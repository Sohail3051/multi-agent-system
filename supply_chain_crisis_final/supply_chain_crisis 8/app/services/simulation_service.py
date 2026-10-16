from __future__ import annotations
from ..database import rows, one
from .metrics_service import calculate_metrics


def _limit(value, count):
    if value == 'all': return count
    try: return min(int(value), count)
    except Exception: return min(2, count)

def generate_candidates(scenario, evidence):
    candidates=[]; ctype=scenario.crisis_type
    if ctype=='supplier':
        for alt in evidence.get('alternatives',[]):
            candidates.append({'candidate_id':f"supplier:{alt['supplier_id']}",'action_type':'alternative_supplier',
              'entity_id':alt['supplier_id'],'entity_name':alt['supplier_name'],'historical_po_count':alt['historical_po_count'],
              'shared_material_count':alt['shared_material_count'],'feasibility':'partially_verified',
              'hard_constraints':['Alternative has historical PO-line relationship to at least one affected material.'],
              'unresolved_checks':['Current capacity','Willingness to accept emergency orders','Current quotation and lead time','Exact material-level overlap for each selected allocation']})
        if not candidates:
            candidates.append({'candidate_id':'supplier:abstain','action_type':'abstain','entity_id':None,'entity_name':'No historically supported alternative found','feasibility':'not_feasible','unresolved_checks':['No alternative supplier relationship verified in historical order lines.']})
        limit=_limit(scenario.max_alternatives,len(candidates))
    elif ctype=='carrier':
        for alt in evidence.get('alternatives',[]):
            candidates.append({'candidate_id':f"carrier:{alt['CarrierID']}",'action_type':'alternative_carrier','entity_id':alt['CarrierID'],
              'entity_name':alt['CarrierName'],'historical_shipments':int(alt['historical_shipments'] or 0),
              'historical_delivered_shipments':int(alt['delivered_shipments'] or 0),'feasibility':'partially_verified',
              'hard_constraints':['Carrier exists in CarriersMaster.'],'unresolved_checks':['Current carrier capacity','Current rate','Transit time and service-level agreement','Route compatibility']})
        if not candidates:
            candidates.append({'candidate_id':'carrier:abstain','action_type':'abstain','entity_id':None,'entity_name':'No alternative carrier listed','feasibility':'not_feasible','unresolved_checks':['No alternative carrier in CarriersMaster.']})
        limit=_limit(scenario.max_alternative_carriers,len(candidates))
    else:
        strategy=scenario.recovery_strategy
        candidates.append({'candidate_id':'warehouse:wait','action_type':'wait_and_monitor','entity_id':scenario.warehouse_id,
          'entity_name':'Wait and monitor the affected warehouse','feasibility':'operationally_descriptive',
          'hard_constraints':['No unverified inventory transfer or capacity claim is made.'],
          'unresolved_checks':['Actual recovery duration','Service impact during disruption']})
        if strategy in {'redirect_shipments','evaluate_transfers'}:
            for alt in evidence.get('alternatives',[]):
                candidates.append({'candidate_id':f"warehouse:{alt['warehouse_id']}",'action_type':'evaluate_alternate_warehouse',
                  'entity_id':alt['warehouse_id'],'entity_name':alt['warehouse_name'],'recorded_inventory_units':alt['recorded_inventory_units'],
                  'feasibility':'candidate_only','hard_constraints':['Alternative warehouse exists in WarehouseMaster.'],
                  'unresolved_checks':['Receiving capacity','Material compatibility','Transfer cost and transit time','Whether the alternative inventory is available for transfer']})
        limit=len(candidates)
    candidates=candidates[:limit]
    for c in candidates:
        c['projected_savings']=None
        c['projected_delay_days']=None
        c['projected_outcome_status']='unknown'
    return candidates

def validate_candidates(scenario, candidates):
    validated=[]; rejected=[]
    for c in candidates:
        ok=True; reasons=[]
        if c.get('action_type')=='alternative_supplier':
            ok=bool(one('SELECT id FROM SupplierMaster WHERE id=?',(c.get('entity_id'),)))
            if not ok: reasons.append('Supplier does not exist in SupplierMaster.')
        elif c.get('action_type')=='alternative_carrier':
            ok=bool(one('SELECT CarrierID FROM CarriersMaster WHERE CarrierID=?',(c.get('entity_id'),)))
            if not ok: reasons.append('Carrier does not exist in CarriersMaster.')
        elif c.get('action_type')=='evaluate_alternate_warehouse':
            ok=bool(one('SELECT id FROM WarehouseMaster WHERE id=?',(c.get('entity_id'),)))
            if not ok: reasons.append('Warehouse does not exist in WarehouseMaster.')
        if ok: validated.append(c)
        else: rejected.append({'candidate':c,'reasons':reasons})
    return validated,rejected

def run_simulation(scenario):
    evidence=calculate_metrics(scenario)
    candidates=generate_candidates(scenario,evidence)
    validated,rejected=validate_candidates(scenario,candidates)
    return {'scenario':scenario.model_dump(mode='json'),'evidence':evidence,'candidates':validated,'rejected_candidates':rejected}
