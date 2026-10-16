from typing import Literal, Optional, Union
from pydantic import BaseModel, Field, ConfigDict, model_validator

CrisisType = Literal["supplier", "carrier", "warehouse"]
Objective = Literal["minimize_cost", "minimize_lead_time", "maximize_reliability", "balanced"]
RiskTolerance = Literal["conservative", "balanced", "aggressive"]

class Preferences(BaseModel):
    objective: Objective = "balanced"
    risk_tolerance: RiskTolerance = "balanced"
    analysis_depth: Literal["standard", "detailed"] = "standard"
    compare_alternatives: Literal["1", "2", "3", "all"] = "3"
    sensitivity_analysis: bool = False
    show_assumptions: bool = True

class ScenarioBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    crisis_type: CrisisType
    duration_days: int = Field(default=7, ge=1, le=365)
    severity: Literal["partial", "complete"] = "partial"
    preferences: Preferences = Field(default_factory=Preferences)

class SupplierScenario(ScenarioBase):
    crisis_type: Literal["supplier"]
    supplier_id: str = Field(min_length=1, max_length=80)
    material_scope: Literal["all", "selected"] = "all"
    material_ids: list[str] = Field(default_factory=list, max_length=50)
    procurement_scope: Literal["eligible", "recent_year"] = "eligible"
    priority_basis: Literal["all_affected", "highest_value", "low_inventory"] = "all_affected"
    sourcing_strategy: Literal["single", "split", "all"] = "all"
    max_alternatives: Literal["1", "2", "all"] = "2"
    @model_validator(mode="after")
    def selected_materials_required(self):
        if self.material_scope == "selected" and not self.material_ids:
            raise ValueError("Select at least one material when material scope is selected.")
        return self

class CarrierScenario(ScenarioBase):
    crisis_type: Literal["carrier"]
    carrier_id: str = Field(min_length=1, max_length=80)
    shipment_scope: Literal["all_eligible", "origin_warehouse"] = "all_eligible"
    origin_warehouse_id: Optional[str] = None
    rerouting_strategy: Literal["single", "split", "all"] = "all"
    max_alternative_carriers: Literal["1", "2", "all"] = "2"
    @model_validator(mode="after")
    def warehouse_required(self):
        if self.shipment_scope == "origin_warehouse" and not self.origin_warehouse_id:
            raise ValueError("Select an origin warehouse for this shipment scope.")
        return self

class WarehouseScenario(ScenarioBase):
    crisis_type: Literal["warehouse"]
    warehouse_id: str = Field(min_length=1, max_length=80)
    inventory_scope: Literal["all", "selected"] = "all"
    material_ids: list[str] = Field(default_factory=list, max_length=50)
    inventory_strategy: Literal["preserve", "prioritize", "evaluate_transfers"] = "preserve"
    recovery_strategy: Literal["wait", "redirect_shipments", "evaluate_transfers"] = "wait"
    @model_validator(mode="after")
    def selected_materials_required(self):
        if self.inventory_scope == "selected" and not self.material_ids:
            raise ValueError("Select at least one material when inventory scope is selected.")
        return self

ScenarioRequest = Union[SupplierScenario, CarrierScenario, WarehouseScenario]

class AnalyzeRequest(BaseModel):
    scenario: dict

class Metric(BaseModel):
    name: str
    value: Union[str, int, float, None]
    unit: str = ""
    status: Literal["calculated", "estimated", "assumption_dependent", "unknown"] = "calculated"
    source: str = ""
    note: Optional[str] = None

class APIError(BaseModel):
    error_code: str
    message: str
    details: Optional[dict] = None
