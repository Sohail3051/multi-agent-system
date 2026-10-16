from __future__ import annotations
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import TypeAdapter, ValidationError
from pathlib import Path
import sqlite3, time
from contextlib import closing
from .config import APP_NAME, DB_PATH, GOOGLE_API_KEY, GEMINI_MODEL
from .database import rows, one
from .schemas import SupplierScenario, CarrierScenario, WarehouseScenario
from .services.metrics_service import calculate_metrics
from .services.simulation_service import run_simulation
from .agents.workflow import run_agents

ROOT=Path(__file__).resolve().parents[1]
app=FastAPI(title=APP_NAME,version='1.0.0',description='Evidence-first multi-agent supply-chain crisis decision support.')
app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')
templates=Jinja2Templates(directory=ROOT/'templates')
def _asset_version():
    # Cache-busting token so edited CSS/JS is not served stale from the browser cache.
    try: return str(int(max((ROOT/'static'/n).stat().st_mtime for n in ('app.js','style.css'))))
    except OSError: return '1'
ASSET_VERSION=_asset_version()
SCENARIO_MODELS={'supplier':SupplierScenario,'carrier':CarrierScenario,'warehouse':WarehouseScenario}

def parse_scenario(payload):
    if not isinstance(payload,dict): raise ValueError('Scenario must be a JSON object.')
    crisis=payload.get('crisis_type')
    model=SCENARIO_MODELS.get(crisis)
    if model is None: raise ValueError("crisis_type must be one of: supplier, carrier, warehouse.")
    return model.model_validate(payload)

def error_response(code,message,details=None,status=400):
    return JSONResponse(status_code=status,content={'error_code':code,'message':message,'details':details})

@app.get('/',response_class=HTMLResponse)
def dashboard(request:Request):
    return templates.TemplateResponse(request=request,name='index.html',context={'app_name':APP_NAME,'asset_version':ASSET_VERSION})

@app.get('/api/options')
def options():
    try:
        return {'suppliers':rows('SELECT id AS value,name AS label,country FROM SupplierMaster ORDER BY name'),
          'carriers':rows('SELECT CarrierID AS value,CarrierName AS label FROM CarriersMaster ORDER BY CarrierName'),
          'warehouses':rows('SELECT id AS value,name AS label,location FROM WarehouseMaster ORDER BY name'),
          'materials':rows('SELECT id AS value,name AS label,category,unitCost FROM RawMaterialsMaster ORDER BY name'),
          'supplier_materials':rows('''SELECT po.SupplierID AS supplier_id,pol.MaterialID AS material_id FROM PurchaseOrders po JOIN PurchaseOrderLines pol ON pol.PO_ID=po.PO_ID GROUP BY po.SupplierID,pol.MaterialID ORDER BY po.SupplierID,pol.MaterialID'''),
          'warehouse_materials':rows('''SELECT WarehouseID AS warehouse_id,MaterialID AS material_id FROM WarehouseInventory GROUP BY WarehouseID,MaterialID ORDER BY WarehouseID,MaterialID'''),
          'database':{'path':str(DB_PATH.name),'read_only':True}}
    except FileNotFoundError as e: return error_response('DATABASE_NOT_READY',str(e),{'hint':'Run scripts/build_database.py using the provided workbook.'},503)

@app.post('/api/metrics')
def metrics(body:dict):
    try:
        scenario=parse_scenario(body.get('scenario',body))
        return calculate_metrics(scenario)
    except ValidationError as e: return error_response('VALIDATION_ERROR','Scenario validation failed.',{'errors':e.errors()},422)
    except ValueError as e: return error_response('INVALID_SCENARIO',str(e),None,422)
    except FileNotFoundError as e: return error_response('DATABASE_NOT_READY',str(e),None,503)
    except sqlite3.Error as e: return error_response('DATABASE_ERROR','A database query failed.',{'type':type(e).__name__},500)

@app.post('/api/analyze')
def analyze(body:dict):
    start=time.time()
    try:
        scenario=parse_scenario(body.get('scenario',body))
        result=run_simulation(scenario)
        payload={'scenario':result['scenario'],'evidence':result['evidence'],'candidates':result['candidates']}
        agent_result=run_agents(payload)
        result.update(agent_result)
        result['elapsed_seconds']=round(time.time()-start,3)
        result['advisory_only']=True
        return result
    except ValidationError as e: return error_response('VALIDATION_ERROR','Scenario validation failed.',{'errors':e.errors()},422)
    except ValueError as e: return error_response('INVALID_SCENARIO',str(e),None,422)
    except FileNotFoundError as e: return error_response('DATABASE_NOT_READY',str(e),None,503)
    except sqlite3.Error as e: return error_response('DATABASE_ERROR','A database query failed.',{'type':type(e).__name__},500)
    except Exception as e: return error_response('ANALYSIS_FAILED','Analysis could not be completed. Previously calculated metrics may still be available from /api/metrics.',{'type':type(e).__name__},500)

@app.post('/api/compare')
def compare(body:dict):
    scenarios=body.get('scenarios')
    if not isinstance(scenarios,list) or not 2<=len(scenarios)<=5:
        return error_response('VALIDATION_ERROR','Provide between 2 and 5 scenario objects in scenarios.',None,422)
    output=[]
    try:
        for index,item in enumerate(scenarios):
            scenario=parse_scenario(item)
            evidence=calculate_metrics(scenario)
            output.append({'scenario_index':index+1,'scenario':scenario.model_dump(mode='json'),'metrics':evidence['metrics'],'selected_entity':evidence['selected_entity']})
        return {'comparisons':output,'note':'Metrics are compared side by side; incompatible units are not combined into a single score.'}
    except ValidationError as e: return error_response('VALIDATION_ERROR','One or more scenarios failed validation.',{'errors':e.errors()},422)
    except ValueError as e: return error_response('INVALID_SCENARIO',str(e),None,422)
    except FileNotFoundError as e: return error_response('DATABASE_NOT_READY',str(e),None,503)
    except sqlite3.Error as e: return error_response('DATABASE_ERROR','A database query failed.',{'type':type(e).__name__},500)

@app.get('/api/health')
def health():
    try:
        with closing(sqlite3.connect(f'file:{DB_PATH}?mode=ro',uri=True)) as conn:
            tables=[r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            return {'status':'ok','database':'connected','database_file_exists':DB_PATH.exists(),'table_count':len(tables),'expected_tables_present':all(t in tables for t in ['PurchaseOrders','PurchaseOrderLines','ShipmentTracking','WarehouseInventory','SupplierMaster','CarriersMaster','WarehouseMaster','RawMaterialsMaster']),'gemini_configured':bool(GOOGLE_API_KEY),'gemini_model':GEMINI_MODEL if GOOGLE_API_KEY else None}
    except Exception as e: return JSONResponse(status_code=503,content={'status':'degraded','database':'unavailable','error_type':type(e).__name__})

@app.get('/api/database-profile')
def database_profile():
    try:
        with closing(sqlite3.connect(f'file:{DB_PATH}?mode=ro',uri=True)) as conn:
            table_names=[r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
            profile={t:conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in table_names}
        return {'tables':profile}
    except Exception as e: return error_response('DATABASE_ERROR','Could not profile the database.',{'type':type(e).__name__},503)
