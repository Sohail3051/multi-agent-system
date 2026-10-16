from __future__ import annotations
from datetime import date, timedelta
from ..database import rows, one


def _metric(name, value, unit="", source="", status="calculated", note=None):
    return {"name": name, "value": value, "unit": unit, "source": source, "status": status, "note": note}

def _round(value, digits=2):
    return round(float(value or 0), digits)

def supplier_metrics(s):
    supplier_id=s.supplier_id
    supplier=one('SELECT id,name,country FROM SupplierMaster WHERE id=?',(supplier_id,))
    if not supplier: raise ValueError("Unknown supplier_id.")
    params=[supplier_id]
    date_clause=" AND lower(trim(COALESCE(po.Status,''))) NOT IN ('delivered','cancelled')"
    if s.procurement_scope == "recent_year":
        cutoff=(date.today()-timedelta(days=365)).isoformat()
        date_clause += " AND date(po.OrderDate) >= date(?)"; params.append(cutoff)
    po_rows=rows(f'''SELECT po.PO_ID, po.Status, po.TotalCost, po.OrderDate
        FROM PurchaseOrders po WHERE po.SupplierID=? {date_clause}''',params)
    po_ids=[r['PO_ID'] for r in po_rows]
    if po_ids:
        placeholders=','.join('?' for _ in po_ids)
        line_rows=rows(f'''SELECT pol.PO_ID, pol.MaterialID, pol.Quantity, pol.UnitCost
          FROM PurchaseOrderLines pol WHERE pol.PO_ID IN ({placeholders})''',po_ids)
    else: line_rows=[]
    material_ids=sorted({r['MaterialID'] for r in line_rows if r['MaterialID']})
    if s.material_scope == 'selected':
        bad=set(s.material_ids)-{r['id'] for r in rows('SELECT id FROM RawMaterialsMaster')}
        if bad: raise ValueError(f"Unknown material ID(s): {', '.join(sorted(bad))}")
        allowed=set(s.material_ids)
        unsupported=allowed-set(material_ids)
        if unsupported: raise ValueError(f"Selected material(s) have no verified purchase-order relationship with supplier {supplier_id}: {', '.join(sorted(unsupported))}")
        line_rows=[r for r in line_rows if r['MaterialID'] in allowed]
        po_ids={r['PO_ID'] for r in line_rows}
        po_rows=[r for r in po_rows if r['PO_ID'] in po_ids]
        material_ids=sorted(allowed)
    affected_po_count=len({r['PO_ID'] for r in line_rows})
    procurement_value=sum(float(r['Quantity'] or 0)*float(r['UnitCost'] or 0) for r in line_rows)
    # Status-level exposure is calculated from the same eligible line set. Counts are distinct
    # orders; line counts and values are aggregated once per line to avoid order double-counting.
    status_by_po={r['PO_ID']: (str(r['Status']) if r['Status'] is not None else '(missing)') for r in po_rows}
    status_exposure={}
    for line in line_rows:
        status=status_by_po.get(line['PO_ID'], '(missing)')
        bucket=status_exposure.setdefault(status, {'order_ids':set(), 'line_count':0, 'line_value':0.0})
        bucket['order_ids'].add(line['PO_ID'])
        bucket['line_count'] += 1
        bucket['line_value'] += float(line['Quantity'] or 0) * float(line['UnitCost'] or 0)
    status_exposure={status:{'orders':len(v['order_ids']),'lines':v['line_count'],'line_value':_round(v['line_value'])} for status,v in sorted(status_exposure.items())}
    # Line value is the relevant exposure; header totals are intentionally omitted from the UI
    # to avoid a second, potentially confusing total that can overlap line-level value.
    inventory=[]
    if material_ids:
        ph=','.join('?' for _ in material_ids)
        inventory=rows(f'''SELECT wi.MaterialID, rm.name AS material_name, SUM(wi.QuantityOnHand) AS quantity_on_hand
          FROM WarehouseInventory wi LEFT JOIN RawMaterialsMaster rm ON rm.id=wi.MaterialID
          WHERE wi.MaterialID IN ({ph}) GROUP BY wi.MaterialID,rm.name''',material_ids)
    if material_ids:
        ph_alt=','.join('?' for _ in material_ids)
        alternatives=rows(f'''SELECT sm.id AS supplier_id, sm.name AS supplier_name, COUNT(DISTINCT po.PO_ID) AS historical_po_count,
            COUNT(DISTINCT pol.MaterialID) AS shared_material_count
          FROM SupplierMaster sm JOIN PurchaseOrders po ON po.SupplierID=sm.id
          JOIN PurchaseOrderLines pol ON pol.PO_ID=po.PO_ID
          WHERE sm.id<>? AND pol.MaterialID IN ({ph_alt})
          GROUP BY sm.id,sm.name ORDER BY shared_material_count DESC,historical_po_count DESC''',[supplier_id,*material_ids])
    else:
        alternatives=[]
    perf=one('SELECT OnTimeDeliveryRate,QualityScore,AverageLeadTime_Days FROM SupplierPerformance WHERE SupplierID=?',(supplier_id,))
    metrics=[_metric('Affected purchase orders',affected_po_count,'orders','PurchaseOrders joined to PurchaseOrderLines using PO_ID'),
      _metric('Affected purchase-order lines',len(line_rows),'lines','PurchaseOrderLines filtered to selected supplier and scope'),
      _metric('Affected line value',_round(procurement_value),'dataset currency units','SUM(Quantity × UnitCost) over affected lines; line-level calculation'),
      _metric('Affected materials',len(material_ids),'materials','Distinct MaterialID values in affected lines'),
      _metric('Recorded inventory for affected materials',sum(float(x['quantity_on_hand'] or 0) for x in inventory),'units','WarehouseInventory summed across warehouses for affected materials','calculated','Aggregate only: stock may be in different warehouses, already committed, inaccessible, or unsuitable for the affected orders. It is not confirmed available-to-promise inventory.')]
    if perf:
        metrics += [_metric('Historical on-time delivery rate',_round(perf['OnTimeDeliveryRate'],4),'rate','SupplierPerformance.OnTimeDeliveryRate','historical','Historical performance only; not a prediction for emergency orders.'),_metric('Historical quality score',perf['QualityScore'],'dataset score','SupplierPerformance.QualityScore','historical','Historical score; scale and current batch quality are not independently verified.'),_metric('Historical average lead time',perf['AverageLeadTime_Days'],'days','SupplierPerformance.AverageLeadTime_Days','historical','Historical average; does not establish that a new order can arrive within the scenario duration.')]
    else: metrics.append(_metric('Supplier performance',None,'','SupplierPerformance','unknown','No performance row found.'))
    material_inventory_note = 'Recorded inventory by affected material: ' + ('; '.join(f"{x.get('material_name') or x.get('MaterialID')}: {float(x.get('quantity_on_hand') or 0):,.0f} units across recorded warehouses" for x in inventory) if inventory else 'no inventory rows found for affected materials') + '. These quantities are not verified available-to-promise stock.'
    return {'crisis_type':'supplier','selected_entity':supplier,'metrics':metrics,'affected_materials':inventory,'alternatives':alternatives,
      'affected_statuses':_counts([r['Status'] for r in po_rows]),'status_exposure':status_exposure,'evidence_notes':['Alternative suppliers are identified only from historical purchase-order relationships to affected materials. Current supplier capacity, willingness, quotations, and lead times are unknown unless explicitly recorded.',material_inventory_note,'Historical on-time delivery, quality, and lead time describe past records; they do not predict emergency-order performance or prove fulfilment within the scenario duration.','Disruption duration and severity are scenario assumptions, not predictions. Delivered and cancelled purchase orders are excluded from the eligible procurement scope. Status-level exposure groups the remaining eligible purchase-order lines by their recorded order status; a Shipped status is not proof of confirmed delivery.']}

def carrier_metrics(s):
    carrier=one('SELECT CarrierID,CarrierName FROM CarriersMaster WHERE CarrierID=?',(s.carrier_id,))
    if not carrier: raise ValueError('Unknown carrier_id.')
    params=[s.carrier_id]; where='WHERE st.CarrierID=?'
    if s.shipment_scope=='origin_warehouse':
        warehouse=one('SELECT id,name,location FROM WarehouseMaster WHERE id=?',(s.origin_warehouse_id,))
        if not warehouse: raise ValueError('Unknown origin_warehouse_id.')
        where+=' AND st.OriginWarehouseID=?'; params.append(s.origin_warehouse_id)
    else: warehouse=None
    ship_rows=rows(f'''SELECT st.ShipmentID,st.Status,st.OriginWarehouseID,st.CarrierID FROM ShipmentTracking st {where}''',params)
    delivered={'delivered','complete','completed'}
    eligible=[r for r in ship_rows if str(r['Status'] or '').strip().lower() not in delivered]
    statuses=_counts([r['Status'] for r in ship_rows])
    alternatives=rows('''SELECT cm.CarrierID,cm.CarrierName,COUNT(st.ShipmentID) AS historical_shipments,
        SUM(CASE WHEN lower(trim(COALESCE(st.Status,''))) IN ('delivered','complete','completed') THEN 1 ELSE 0 END) AS delivered_shipments
        FROM CarriersMaster cm LEFT JOIN ShipmentTracking st ON st.CarrierID=cm.CarrierID
        WHERE cm.CarrierID<>? GROUP BY cm.CarrierID,cm.CarrierName ORDER BY historical_shipments DESC''',(s.carrier_id,))
    # Verified bridge: FulfillmentStatus links shipment to sales order. Counts only; no causal delay claim.
    ids=[r['ShipmentID'] for r in eligible]
    linked_orders=0
    if ids:
        ph=','.join('?' for _ in ids)
        linked=one(f'SELECT COUNT(DISTINCT SalesOrderID) AS n FROM FulfillmentStatus WHERE ShipmentID IN ({ph})',ids)
        linked_orders=int(linked['n'] or 0)
    metrics=[_metric('Shipments associated with carrier',len(ship_rows),'shipments','ShipmentTracking.CarrierID'),
      _metric('Eligible rerouting candidates',len(eligible),'shipments','ShipmentTracking records excluding delivered/completed statuses'),
      _metric('Linked customer orders',linked_orders,'orders','Distinct SalesOrderID values linked through FulfillmentStatus; association only, not proof of delay'),
      _metric('Affected origin warehouses',len({r['OriginWarehouseID'] for r in ship_rows if r['OriginWarehouseID']}),'warehouses','Distinct OriginWarehouseID values in selected shipments')]
    return {'crisis_type':'carrier','selected_entity':carrier,'selected_warehouse':warehouse,'metrics':metrics,'alternatives':alternatives,
      'affected_statuses':statuses,'evidence_notes':['Alternative carriers reflect historical use in ShipmentTracking; their current capacity, rates, and transit times are unknown.','No exact delay or cost savings are estimated because the workbook does not establish those values.']}

def warehouse_metrics(s):
    warehouse=one('SELECT id,name,location FROM WarehouseMaster WHERE id=?',(s.warehouse_id,))
    if not warehouse: raise ValueError('Unknown warehouse_id.')
    params=[s.warehouse_id]; inv_where='WHERE wi.WarehouseID=?'
    if s.inventory_scope=='selected':
        bad=set(s.material_ids)-{r['id'] for r in rows('SELECT id FROM RawMaterialsMaster')}
        if bad: raise ValueError(f"Unknown material ID(s): {', '.join(sorted(bad))}")
        if not s.material_ids: raise ValueError('Select at least one material.')
        ph=','.join('?' for _ in s.material_ids); inv_where+=f' AND wi.MaterialID IN ({ph})'; params.extend(s.material_ids)
    inventory=rows(f'''SELECT wi.MaterialID,rm.name AS material_name,wi.QuantityOnHand,wi.LastStocktakeDate
      FROM WarehouseInventory wi LEFT JOIN RawMaterialsMaster rm ON rm.id=wi.MaterialID {inv_where} ORDER BY wi.MaterialID''',params)
    shipments=rows('SELECT ShipmentID,Status FROM ShipmentTracking WHERE OriginWarehouseID=?',(s.warehouse_id,))
    alternatives=rows('''SELECT wm.id AS warehouse_id,wm.name AS warehouse_name,wm.location,
      COUNT(DISTINCT wi.MaterialID) AS materials_with_recorded_inventory,
      COALESCE(SUM(wi.QuantityOnHand),0) AS recorded_inventory_units
      FROM WarehouseMaster wm LEFT JOIN WarehouseInventory wi ON wi.WarehouseID=wm.id
      WHERE wm.id<>? GROUP BY wm.id,wm.name,wm.location ORDER BY recorded_inventory_units DESC''',(s.warehouse_id,))
    metrics=[_metric('Recorded inventory',_round(sum(float(r['QuantityOnHand'] or 0) for r in inventory),0),'units','WarehouseInventory.QuantityOnHand at selected warehouse'),
      _metric('Originating shipments',len(shipments),'shipments','ShipmentTracking.OriginWarehouseID'),
      _metric('Delivered originating shipments',sum(1 for r in shipments if str(r['Status'] or '').strip().lower() in {'delivered','complete','completed'}),'shipments','ShipmentTracking status values')]
    return {'crisis_type':'warehouse','selected_entity':warehouse,'metrics':metrics,'inventory':inventory,'alternatives':alternatives,
      'affected_statuses':_counts([r['Status'] for r in shipments]),'evidence_notes':['Inventory quantities are recorded stock, not verified transferable stock.','Warehouse capacity, receiving capacity, material compatibility, and transfer cost/time are not present as verified facts.']}

def _counts(values):
    result={}
    for value in values:
        key=str(value) if value is not None else '(missing)'; result[key]=result.get(key,0)+1
    return dict(sorted(result.items(),key=lambda x:x[0]))

def calculate_metrics(scenario):
    if scenario.crisis_type=='supplier': return supplier_metrics(scenario)
    if scenario.crisis_type=='carrier': return carrier_metrics(scenario)
    return warehouse_metrics(scenario)
