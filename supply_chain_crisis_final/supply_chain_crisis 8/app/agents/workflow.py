"""Four-agent advisory workflow. Deterministic fallback works without an API key.
Gemini-backed nodes are used when GOOGLE_API_KEY and optional dependencies are available.
"""
from __future__ import annotations
import json, os, logging, re, time
from typing import TypedDict, Any
from ..config import GOOGLE_API_KEY, GEMINI_MODEL
from .schemas import AgentFinding, DecisionOutput

logger = logging.getLogger(__name__)

class WorkflowState(TypedDict, total=False):
    payload: dict
    operations: dict
    risk: dict
    strategy: dict
    decision: dict
    mode: str
    errors: list[str]

def _expected_json_contract(role: str) -> str:
    """Stable output contracts plus role-specific responsibilities prevent duplicate agent reports."""
    if role == 'Decision Agent':
        return ('Return exactly one JSON object with keys: selected_candidate_id (string exactly matching one supplied candidate_id, or null), '
                'recommendation (string), rationale (array of strings), tradeoffs (array of strings), assumptions (array of strings), '
                'unresolved_checks (array of strings), abstained (boolean). Never return a supplier name in place of candidate_id. '
                'A selected candidate with unresolved checks is for verification only: explicitly say not to place/reallocate orders until checks pass.')
    contracts = {
        'Operations Agent': ('Your unique remit is execution-level operations: identify affected orders/lines/materials, inventory location/availability limits, '
            'order status, bottlenecks, and what an operations team should verify next. Use calculated metrics as supplied; do not recompute. '
            'Do not write a general risk report or rank suppliers. Findings must be operationally specific and refer to provided evidence.'),
        'Risk Agent': ('Your unique remit is risk analysis, not a recap of counts. Identify failure modes, uncertainty, consequence pathways, and controls/mitigations. '
            'Explicitly assess the risk of assuming aggregate inventory is usable, historical lead time meets the scenario deadline, or past relationships imply emergency capacity. '
            'Do not merely restate the Operations summary. Do not invent probability scores or numeric severity.'),
        'Strategy Agent': ('Your unique remit is comparing the supplied candidate options against the stated objective. Compare their actual supplied historical signals, '
            'feasibility labels, and unresolved checks; call out ties or weak evidence. State why the ranking does or does not distinguish candidates. '
            'Do not repeat the Operations exposure recap or generic risk list, and do not invent savings, delays, or current feasibility.'),
    }
    return ('Return exactly one JSON object with keys: summary (non-empty string), findings (array of strings), '
            'evidence_refs (array of strings), caveats (array of strings). Do not rename these keys. '
            'Use an empty array when there is no supported item; do not omit summary. ' + contracts.get(role, 'Stay within your assigned role.'))


def _normalize_agent_keys(raw: dict | None, role: str) -> dict:
    """Map common model-produced aliases onto our stable UI contract."""
    if not isinstance(raw, dict):
        return {}
    data = dict(raw)
    aliases = {
        'Operations Agent': ('operations_summary', 'operational_summary', 'assessment', 'analysis', 'overview'),
        'Risk Agent': ('risk_summary', 'risk_assessment', 'assessment', 'analysis', 'overview'),
        'Strategy Agent': ('strategy_summary', 'strategic_summary', 'assessment', 'analysis', 'overview'),
    }
    if not isinstance(data.get('summary'), str) or not data.get('summary', '').strip():
        for key in aliases.get(role, ()):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                data['summary'] = value
                break
    # Some models return lists as a single paragraph or use alternate common labels.
    for target, options in {
        'findings': ('key_findings', 'observations', 'risks', 'recommendations'),
        'evidence_refs': ('sources', 'evidence', 'data_sources'),
        'caveats': ('limitations', 'uncertainties', 'assumptions'),
    }.items():
        if not data.get(target):
            for key in options:
                if data.get(key) is not None:
                    data[target] = data[key]
                    break
    return data


def _gemini_json(role: str, instructions: str, payload: dict, errors: list[str] | None = None) -> dict | None:
    if not GOOGLE_API_KEY:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage, SystemMessage
        # Retry only transient provider/service failures. Gemini can return 429/503
        # during short demand spikes; a single failure should not immediately discard
        # an otherwise usable agent report. Keep retries bounded to avoid long hangs.
        model=ChatGoogleGenerativeAI(model=GEMINI_MODEL,google_api_key=GOOGLE_API_KEY,temperature=0,timeout=25,max_retries=0)
        system_prompt=(f"You are the {role} in a supply-chain decision-support system. {instructions} "
                       f"{_expected_json_contract(role)} Never invent evidence, numerical metrics, capacity, prices, or time savings. "
                       'Return valid JSON only: no Markdown fences, no preamble, no extra keys.')
        messages=[SystemMessage(content=system_prompt),HumanMessage(content=json.dumps(payload,default=str)[:24000])]
        result = None
        for attempt in range(3):
            try:
                result = model.invoke(messages)
                break
            except Exception as call_exc:
                error_text = f"{type(call_exc).__name__}: {call_exc}".lower()
                transient = any(token in error_text for token in (
                    '503', 'unavailable', '429', 'resource_exhausted', 'too many requests',
                    '502', 'bad gateway', '504', 'gateway timeout', '500 internal'
                ))
                if not transient or attempt >= 2:
                    raise
                delay = 1.25 * (2 ** attempt)
                logger.warning('%s Gemini transient failure on attempt %s/3; retrying in %.2fs', role, attempt + 1, delay)
                time.sleep(delay)
        if result is None:
            return None
        content=result.content
        if isinstance(content, list):
            content=''.join(str(x.get('text','')) if isinstance(x,dict) else str(x) for x in content)
        if isinstance(content, dict):
            return content
        content=str(content).strip()
        # Gemini sometimes wraps otherwise-valid JSON in Markdown fences or adds a short preamble.
        content=re.sub(r'^```(?:json)?\s*', '', content, flags=re.IGNORECASE)
        content=re.sub(r'\s*```$', '', content)
        try:
            parsed=json.loads(content)
        except json.JSONDecodeError:
            start, end = content.find('{'), content.rfind('}')
            if start < 0 or end <= start:
                raise
            parsed=json.loads(content[start:end+1])
        return parsed if isinstance(parsed,dict) else None
    except Exception as exc:
        # Do not leak API keys or raw provider payloads; keep a useful error for the dashboard.
        message = f"{role} Gemini call failed ({type(exc).__name__}): {str(exc)[:220]}"
        logger.warning(message)
        if errors is not None:
            errors.append(message)
        return None

def _merge_agent_response(raw: dict | None, fallback: dict, role: str, errors: list[str] | None = None) -> dict:
    """Keep useful deterministic content when Gemini returns partial/empty fields.

    A syntactically valid JSON object is not necessarily a usable agent report. In
    particular, {"summary": null} used to pass through and render as "summary
    unavailable" while silently dropping the deterministic findings.
    """
    raw = _normalize_agent_keys(raw, role)
    summary = raw.get('summary')
    if not isinstance(summary, str) or not summary.strip():
        summary = fallback['summary']
        source = 'deterministic_fallback'
        if errors is not None:
            keys = ', '.join(sorted(str(k) for k in raw.keys())) or 'no JSON keys'
            errors.append(f'{role} returned no usable summary (response keys: {keys}); deterministic report used.')
    else:
        source = 'gemini'

    used_fallback_fields = source != 'gemini'
    def usable_list(key):
        nonlocal used_fallback_fields
        values = _as_text_list(raw.get(key))
        if values:
            return values
        fallback_values = list(fallback.get(key, []))
        if fallback_values:
            used_fallback_fields = True
        return fallback_values

    findings = usable_list('findings')
    evidence_refs = usable_list('evidence_refs')
    caveats = usable_list('caveats')
    if source == 'gemini' and used_fallback_fields:
        source = 'mixed'
    return {
        'summary': summary.strip(),
        'findings': findings,
        'evidence_refs': evidence_refs,
        'caveats': caveats,
        'source': source,
    }


def operations_agent(payload: dict, errors: list[str] | None = None) -> dict:
    e=payload['evidence']; scenario=payload['scenario']
    summary=f"Operations assessment: a {scenario['severity']} {scenario['crisis_type']} disruption lasting {scenario['duration_days']} days is being assessed. The exposure figures below are calculated from the local supply-chain database; they describe recorded exposure, not confirmed shortages or service failures."
    findings=[f"{m['name']}: {m['value']} {m.get('unit','')} (source: {m.get('source','')})." for m in e.get('metrics',[])]
    fallback={'summary':summary,'findings':findings,'evidence_refs':[m.get('source','') for m in e.get('metrics',[]) if m.get('source')],'caveats':e.get('evidence_notes',[])}
    raw = _gemini_json('Operations Agent','Describe the operational footprint, order/material flow, stock-location caveats, and concrete next operational checks. Preserve exact metrics; do not recompute them.',payload,errors)
    return _merge_agent_response(raw, fallback, 'Operations Agent', errors)

def risk_agent(payload: dict, errors: list[str] | None = None) -> dict:
    e=payload['evidence']; candidates=payload.get('candidates',[])
    fallback={'summary':'Risk assessment: the principal risks are an unverified supply gap, false confidence from aggregated stock, and selecting a candidate whose current capacity or delivery timing is unknown. The scenario does not establish that a shortage has occurred or that an alternative can meet the deadline.',
      'findings':['Inventory risk: recorded stock may be in another warehouse, committed to existing demand, inaccessible, or unsuitable; confirm available-to-promise quantity by material and location.',
        'Deadline risk: historical average lead time is not an emergency quote or a confirmed transit commitment; compare a supplier-confirmed arrival date with the scenario window.',
        'Continuity risk: historical material overlap does not establish current production capacity, willingness, quality for the current batch, or order acceptance.',
        'Commercial risk: current price, minimum order quantity, expedite charges, and cancellation/reallocation terms are not established by the supplied evidence.',
        'Control: require material-level quantity confirmation, written quotation, confirmed ship/arrival date, and an authorized procurement approval before reallocating orders.'],
      'evidence_refs':[m.get('source','') for m in e.get('metrics',[]) if m.get('source')],
      'caveats':['Do not interpret recorded inventory or historical supplier performance as a probability of successful emergency fulfilment.','Disruption severity and duration are user-supplied assumptions; no quantified loss probability or savings estimate is supported.']}
    raw = _gemini_json('Risk Agent','Produce a risk register in prose findings: failure mode, why current evidence does/does not support it, and the control required. Focus on uncertainties and mitigations, not repeating the exposure counts. IMPORTANT: a historical average lead time longer than the scenario window is a warning signal, not proof that every expedited order is impossible. State that only a current written delivery commitment can establish feasibility. Do not claim a historical average mathematically proves the scenario cannot be met.',payload,errors)
    report = _merge_agent_response(raw, fallback, 'Risk Agent', errors)
    # Correct a common overstatement in generated risk reports: historical averages are not guarantees.
    corrected = []
    changed = False
    for finding in report.get('findings', []):
        text = finding
        for phrase in ('directly violates the timeline constraints', 'proves the timeline cannot be met', 'makes seven-day delivery impossible', 'makes 7-day delivery impossible'):
            if phrase in text.lower():
                text = re.sub(re.escape(phrase), 'raises a serious feasibility concern against the scenario window but does not prove expedited delivery is impossible', text, flags=re.IGNORECASE)
                changed = True
        corrected.append(text)
    report['findings'] = corrected
    required = 'A historical average lead time is a warning signal, not proof every emergency order will take that long; require a current written arrival commitment against the scenario deadline.'
    if not any('historical average lead time is a warning signal' in x.lower() for x in report.get('caveats', [])):
        report.setdefault('caveats', []).append(required)
        changed = True
    if changed and report.get('source') == 'gemini':
        report['source'] = 'mixed'
    return report

def strategy_agent(payload: dict, errors: list[str] | None = None) -> dict:
    scenario=payload['scenario']; candidates=payload.get('candidates',[])
    objective=scenario.get('preferences',{}).get('objective','balanced')
    ordered=sorted(candidates,key=lambda c:(int(c.get('shared_material_count',0)),int(c.get('historical_po_count',0))),reverse=True)
    comparison=[]
    for c in ordered:
        if c.get('action_type')=='alternative_supplier':
            comparison.append(f"{c.get('entity_name')} ({c.get('candidate_id')}): {c.get('shared_material_count',0)} historically shared materials; {c.get('historical_po_count',0)} historical POs; feasibility={c.get('feasibility')}.")
        else:
            comparison.append(f"{c.get('entity_name')} ({c.get('candidate_id')}): feasibility={c.get('feasibility')}; unresolved checks: {', '.join(c.get('unresolved_checks',[])) or 'none recorded'}.")
    fallback={'summary':f"Strategy assessment: compared {len(candidates)} validated candidate action(s) against the {objective} objective. The ranking uses historical relationship counts only and cannot establish current operational superiority.",
      'findings':comparison or ['No validated alternative candidates were supplied; gather operational data before selecting an action.'],
      'evidence_refs':[c.get('candidate_id','') for c in candidates if c.get('candidate_id')],
      'caveats':['Historical PO/material overlap is a screening signal, not confirmation of current quantity, capacity, price, willingness, or delivery within the scenario window.']}
    raw = _gemini_json('Strategy Agent','Compare candidates on their actual evidence fields. Explicitly report whether the objective meaningfully differentiates candidates, and separate ranking evidence from feasibility evidence. Read each numeric value exactly: do not call historical PO counts identical unless the counts are numerically equal. If shared-material counts tie but PO counts differ, state the exact difference and do not treat it as proof of current operational superiority.',payload,errors)
    report = _merge_agent_response(raw, fallback, 'Strategy Agent', errors)
    # Always attach a deterministic, database-backed comparison so a fluent but inaccurate model sentence cannot be the only comparison.
    supplier_candidates = [c for c in candidates if c.get('action_type') == 'alternative_supplier']
    if supplier_candidates:
        verified = []
        for c in sorted(supplier_candidates, key=lambda x: (int(x.get('shared_material_count', 0)), int(x.get('historical_po_count', 0))), reverse=True):
            verified.append(f"Verified comparison: {c.get('entity_name')} ({c.get('candidate_id')}) has {int(c.get('shared_material_count', 0))} historically shared material(s), {int(c.get('historical_po_count', 0)):,} historical purchase orders, and feasibility status {c.get('feasibility', 'unknown')}.")
        top = max((int(c.get('shared_material_count', 0)), int(c.get('historical_po_count', 0))) for c in supplier_candidates)
        top_candidates = [c for c in supplier_candidates if (int(c.get('shared_material_count', 0)), int(c.get('historical_po_count', 0))) == top]
        if len(top_candidates) > 1:
            verified.append(f"The top historical signal is tied across {len(top_candidates)} candidates; this does not establish current operational superiority.")
        elif len(supplier_candidates) > 1:
            second = sorted(supplier_candidates, key=lambda x: (int(x.get('shared_material_count', 0)), int(x.get('historical_po_count', 0))), reverse=True)[1]
            diff = int(top_candidates[0].get('historical_po_count', 0)) - int(second.get('historical_po_count', 0))
            if int(top_candidates[0].get('shared_material_count', 0)) == int(second.get('shared_material_count', 0)):
                verified.append(f"The leading candidate has {diff:,} more historical purchase order(s) than the next candidate with the same shared-material count; this small historical difference is not proof of better current capacity, price, or delivery.")
        report_findings = list(report.get('findings', []))
        # Remove contradictory generic claims when exact verified comparison is available.
        contradictory = ('identical historical relationship metrics', 'historical metrics are identical', 'exact same historical metrics')
        cleaned = [f for f in report_findings if not any(token in f.lower() for token in contradictory)]
        for item in verified:
            if item not in cleaned:
                cleaned.append(item)
        report['findings'] = cleaned
        if report.get('source') == 'gemini':
            report['source'] = 'mixed'
    return report

def decision_agent(payload: dict, errors: list[str] | None = None) -> dict:
    candidates=payload.get('candidates',[]); scenario=payload['scenario']; objective=scenario.get('preferences',{}).get('objective','balanced')
    # 'abstain' placeholders mean no supported alternative exists; they must never be recommended.
    actionable=[c for c in candidates if c.get('action_type')!='abstain']
    if not actionable:
        placeholder_checks=[x for c in candidates for x in c.get('unresolved_checks',[])]
        fallback={'selected_candidate_id':None,'recommendation':'No supportable candidate action is available. Gather additional operational data before deciding.',
          'rationale':['No candidate passed validation.' if not candidates else 'No historically supported alternative was found in the data.'],'tradeoffs':[],'assumptions':[],
          'unresolved_checks':placeholder_checks or ['Identify verified alternative entities and their current capacity/constraints.'],'abstained':True}
    else:
        # Deterministic tie-breaking by historical evidence count; do not present it as optimization.
        def key(c):
            if c.get('action_type')=='alternative_supplier': return (int(c.get('shared_material_count',0)),int(c.get('historical_po_count',0)))
            if c.get('action_type')=='alternative_carrier': return (int(c.get('historical_delivered_shipments',0)),int(c.get('historical_shipments',0)))
            return (0,0)
        ranked=sorted(actionable,key=key,reverse=True)
        selected=ranked[0]
        if selected.get('action_type')=='wait_and_monitor':
            recommendation='Use a wait-and-monitor posture while verifying recovery conditions; this is not an execution instruction.'
        else:
            recommendation=f"Prioritize {selected['entity_name']} for operational verification; this is not authorization to execute the action."
        selected_key=key(selected)
        tied=[c for c in ranked if key(c)==selected_key]
        rationale=[f"The candidate exists in the verified dataset and is marked {selected.get('feasibility','candidate_only')}.", f"Objective: {objective}. Historical evidence is a screening signal, not a guarantee of future performance."]
        if selected.get('action_type')=='alternative_supplier':
            rationale.append(f"Ranking signal: {selected.get('shared_material_count',0)} historically shared material(s) and {selected.get('historical_po_count',0)} historical purchase order(s).")
        elif selected.get('action_type')=='alternative_carrier':
            rationale.append(f"Ranking signal: {selected.get('historical_delivered_shipments',0)} historically delivered shipment(s) out of {selected.get('historical_shipments',0)} recorded shipment(s).")
        if len(tied)>1:
            rationale.append(f"The top historical ranking is tied across {len(tied)} candidates ({', '.join(c.get('entity_name','unnamed candidate') for c in tied)}); this selection is a deterministic display tie-break, not evidence that {selected.get('entity_name')} is superior.")
        elif len(ranked)>1:
            second=ranked[1]
            rationale.append(f"The selected candidate ranks ahead on the available historical count signal; no current operational advantage is established.")
        fallback={'selected_candidate_id':selected['candidate_id'],'recommendation':recommendation,
          'rationale':rationale,
          'tradeoffs':['Historical relationships do not establish present capacity, pricing, transit time, or willingness.','The candidate cannot be treated as operationally feasible until unresolved checks are completed.'],
          'assumptions':[f"Scenario duration: {scenario.get('duration_days')} days; severity: {scenario.get('severity')} (user-supplied)."],
          'unresolved_checks':selected.get('unresolved_checks',[]),'abstained':False}
    result=_gemini_json('Decision Agent','Select only one candidate_id from the supplied validated candidates, or abstain. Preserve uncertainty and list unresolved checks.',payload,errors)
    if result:
        allowed={c['candidate_id'] for c in candidates}
        if result.get('selected_candidate_id') not in allowed and result.get('selected_candidate_id') is not None: result=None
    return result or fallback

def _as_text_list(value):
    """Coerce common LLM list-shape mistakes into stable, display-safe strings."""
    if value is None:
        return []
    if isinstance(value, str):
        value=value.strip()
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if item is not None and str(item).strip()]
    return [str(value).strip()]

def _validate_agent(data, role='Agent', errors=None):
    data = data if isinstance(data, dict) else {}
    normalized = {
        'summary': str(data.get('summary') or {'Operations Agent': 'Operations assessment: calculated exposure metrics are available, but the agent response was incomplete. Use the recorded counts as exposure indicators, not proof of shortage.', 'Risk Agent': 'Risk assessment: current capacity, emergency willingness, quotations, and lead times remain unverified; do not infer emergency feasibility from historical performance.', 'Strategy Agent': 'Strategy assessment: compare candidates using the recorded historical signals only. Where signals tie, no superiority is established and operational verification is required.'}.get(role, f'{role} deterministic report is being used.')),
        'findings': _as_text_list(data.get('findings')),
        'evidence_refs': _as_text_list(data.get('evidence_refs')),
        'caveats': _as_text_list(data.get('caveats')),
        'source': data.get('source') if data.get('source') in {'gemini', 'deterministic_fallback', 'mixed'} else 'deterministic_fallback',
    }
    try:
        return AgentFinding.model_validate(normalized).model_dump()
    except Exception as exc:
        logger.warning('%s output schema validation failed: %s', role, type(exc).__name__)
        if errors is not None:
            errors.append(f'{role} output schema validation failed ({type(exc).__name__}); safe display normalization used.')
        return AgentFinding(summary=normalized['summary'], findings=normalized['findings'], evidence_refs=normalized['evidence_refs'], caveats=['Agent output was normalized because its fields did not match the expected schema.'], source='deterministic_fallback').model_dump()

def _normalize_decision(data, candidates, errors=None):
    """Normalize Gemini's flexible JSON into the UI/API contract without inventing evidence."""
    data = dict(data) if isinstance(data, dict) else {}
    # Accommodate common naming variations while keeping candidate IDs strictly validated.
    if 'selected_candidate_id' not in data:
        for alias in ('candidate_id', 'selected_candidate', 'selected_supplier_id', 'selected_carrier_id'):
            if data.get(alias) is not None:
                data['selected_candidate_id'] = data[alias]
                break
    if not isinstance(data.get('recommendation'), str) or not data.get('recommendation', '').strip():
        for alias in ('recommendation_text', 'decision', 'selected_action', 'action_summary'):
            value = data.get(alias)
            if isinstance(value, str) and value.strip():
                data['recommendation'] = value
                break
    for target, aliases in {
        'rationale': ('reasons', 'reasoning', 'justification'),
        'tradeoffs': ('risks', 'trade_offs'),
        'assumptions': ('scenario_assumptions',),
        'unresolved_checks': ('still_to_verify', 'verification_needed', 'open_checks'),
    }.items():
        if not data.get(target):
            for alias in aliases:
                if data.get(alias) is not None:
                    data[target] = data[alias]
                    break
    allowed = {c.get('candidate_id'): c for c in candidates if c.get('candidate_id')}
    selected = data.get('selected_candidate_id')
    # If the model returned an exact candidate display name, resolve it only when unique.
    if isinstance(selected, str) and selected not in allowed:
        matches = [c.get('candidate_id') for c in candidates if c.get('entity_name', '').casefold() == selected.casefold()]
        if len(matches) == 1:
            selected = matches[0]
    if selected in ('', 'null', 'None', 'none', 'abstain'):
        selected = None
    if selected is not None and selected not in allowed:
        if errors is not None:
            errors.append('Decision Agent selected an unknown candidate; deterministic decision used instead.')
        return None
    if selected is not None and allowed[selected].get('action_type') == 'abstain':
        selected = None
    recommendation = data.get('recommendation')
    if not isinstance(recommendation, str) or not recommendation.strip():
        if errors is not None:
            errors.append('Decision Agent omitted a usable recommendation; deterministic decision used instead.')
        return None
    normalized = {
        'selected_candidate_id': selected,
        'recommendation': recommendation.strip(),
        'rationale': _as_text_list(data.get('rationale')),
        'tradeoffs': _as_text_list(data.get('tradeoffs')),
        'assumptions': _as_text_list(data.get('assumptions')),
        'unresolved_checks': _as_text_list(data.get('unresolved_checks')),
        'abstained': bool(data.get('abstained', selected is None)) or selected is None,
    }
    try:
        return DecisionOutput.model_validate(normalized).model_dump()
    except Exception as exc:
        logger.warning('Decision output schema validation failed: %s', type(exc).__name__)
        if errors is not None:
            errors.append(f'Decision output schema validation failed ({type(exc).__name__}); deterministic decision used instead.')
        return None

def _enforce_conditional_execution_boundary(decision: dict, candidates: list[dict], duration_days: int | None = None) -> dict:
    """A model may prioritize verification, but cannot turn partially verified evidence into execution authorization."""
    result = dict(decision)
    selected_id = result.get('selected_candidate_id')
    selected = next((c for c in candidates if c.get('candidate_id') == selected_id), None)
    if not selected or result.get('abstained'):
        return result
    unresolved = _as_text_list(selected.get('unresolved_checks'))
    feasibility = str(selected.get('feasibility', '')).lower()
    if unresolved or feasibility in {'partially_verified', 'candidate_only'}:
        name = selected.get('entity_name') or 'the selected candidate'
        window = f" within the {duration_days}-day scenario window" if duration_days else ''
        result['recommendation'] = (f"Prioritize {name} for operational verification only. Do not place or reallocate orders until current capacity, willingness, material-level quantity, quotation, and confirmed delivery timing{window} are verified and procurement approval is obtained.")
        rationale = _as_text_list(result.get('rationale'))
        boundary = 'This is a verification priority, not an instruction to execute procurement; unresolved feasibility checks remain open.'
        if boundary not in rationale:
            rationale.append(boundary)
        objective_note = 'The selected objective is a user preference; the available data does not contain a validated objective-specific score, so historical counts alone do not prove this is the best balanced choice.'
        if objective_note not in rationale:
            rationale.append(objective_note)
        result['rationale'] = rationale
        tradeoffs = _as_text_list(result.get('tradeoffs'))
        if 'Proceeding before verification could create an unfulfilled order or additional cost.' not in tradeoffs:
            tradeoffs.append('Proceeding before verification could create an unfulfilled order or additional cost.')
        result['tradeoffs'] = tradeoffs
        checks = _as_text_list(result.get('unresolved_checks'))
        for check in unresolved:
            if check not in checks:
                checks.append(check)
        mandatory_checks = [
            'Confirm available-to-promise inventory by material and warehouse; aggregate stock is not proof of usable inventory.',
            'Confirm exact material-level quantities and compatibility for each proposed allocation.',
            'Obtain current written quotation and explicit acceptance of the emergency order.',
        ]
        for check in mandatory_checks:
            if not any(check.casefold() == existing.casefold() for existing in checks):
                checks.append(check)
        if duration_days and not any(f'{duration_days}-day scenario window' in c.lower() for c in checks):
            checks.append(f'Confirm delivery can meet the {duration_days}-day scenario window.')
        result['unresolved_checks'] = checks
        result['abstained'] = False
    return result


def run_agents(payload: dict) -> dict:
    errors=[]; gemini_errors=[]; mode='gemini' if GOOGLE_API_KEY else 'deterministic_fallback'
    try:
        from langgraph.graph import StateGraph, END
        def op(state): return {'operations':operations_agent(state['payload'],gemini_errors)}
        def risk(state): return {'risk':risk_agent({**state['payload'],'operations':state.get('operations',{})},gemini_errors)}
        def strategy(state): return {'strategy':strategy_agent({**state['payload'],'operations':state.get('operations',{}),'risk':state.get('risk',{})},gemini_errors)}
        def decision(state): return {'decision':decision_agent({**state['payload'],'operations':state.get('operations',{}),'risk':state.get('risk',{}),'strategy':state.get('strategy',{})},gemini_errors)}
        graph=StateGraph(WorkflowState); graph.add_node('operations',op); graph.add_node('risk',risk); graph.add_node('strategy',strategy); graph.add_node('decision',decision)
        graph.set_entry_point('operations'); graph.add_edge('operations','risk'); graph.add_edge('risk','strategy'); graph.add_edge('strategy','decision'); graph.add_edge('decision',END)
        result=graph.compile().invoke({'payload':payload,'mode':mode,'errors':[]})
    except Exception as exc:
        errors.append(f'LangGraph orchestration unavailable; sequential fallback used ({type(exc).__name__}).')
        result={'operations':operations_agent(payload,gemini_errors),'risk':risk_agent(payload,gemini_errors),'strategy':strategy_agent(payload,gemini_errors),'decision':decision_agent(payload,gemini_errors)}

    candidates=payload.get('candidates', [])
    decision=_normalize_decision(result.get('decision', {}), candidates, errors)
    if decision is not None:
        decision = _enforce_conditional_execution_boundary(decision, candidates, payload.get('scenario', {}).get('duration_days'))
    decision_was_gemini = decision is not None
    if decision is None:
        # Use the deterministic, candidate-backed selector rather than returning a blank card.
        fallback_payload={**payload, 'disable_gemini_for_decision': True}
        candidates_actionable=[c for c in candidates if c.get('action_type') != 'abstain']
        if not candidates_actionable:
            checks=[x for c in candidates for x in c.get('unresolved_checks', [])]
            decision={'selected_candidate_id':None,
              'recommendation':'No supportable candidate action is available. Gather additional operational data before deciding.',
              'rationale':['No historically supported alternative was found in the supplied candidate set.'],
              'tradeoffs':[], 'assumptions':[],
              'unresolved_checks':checks or ['Identify verified alternative entities and their current capacity/constraints.'],
              'abstained':True}
        else:
            def key(c):
                if c.get('action_type')=='alternative_supplier': return (int(c.get('shared_material_count',0)),int(c.get('historical_po_count',0)))
                if c.get('action_type')=='alternative_carrier': return (int(c.get('historical_delivered_shipments',0)),int(c.get('historical_shipments',0)))
                return (0,0)
            selected=sorted(candidates_actionable,key=key,reverse=True)[0]
            recommendation=(f"Use a wait-and-monitor posture while verifying recovery conditions; this is not an execution instruction."
                if selected.get('action_type')=='wait_and_monitor' else
                f"Prioritize {selected.get('entity_name','the selected candidate')} for operational verification; this is not authorization to execute the action.")
            selected_key=key(selected)
            tied=[c for c in sorted(candidates_actionable,key=key,reverse=True) if key(c)==selected_key]
            rationale=[f"The candidate exists in the verified dataset and is marked {selected.get('feasibility','candidate_only')}.", 'Historical evidence is a screening signal, not a guarantee of future performance.']
            if selected.get('action_type')=='alternative_supplier':
                rationale.append(f"Ranking signal: {selected.get('shared_material_count',0)} historically shared material(s) and {selected.get('historical_po_count',0)} historical purchase order(s).")
            elif selected.get('action_type')=='alternative_carrier':
                rationale.append(f"Ranking signal: {selected.get('historical_delivered_shipments',0)} historically delivered shipment(s) out of {selected.get('historical_shipments',0)} recorded shipment(s).")
            if len(tied)>1:
                rationale.append(f"The top historical ranking is tied across {len(tied)} candidates ({', '.join(c.get('entity_name','unnamed candidate') for c in tied)}); this is a deterministic display tie-break, not evidence that {selected.get('entity_name')} is superior.")
            elif len(candidates_actionable)>1:
                rationale.append('The selected candidate ranks ahead on the available historical count signal; no current operational advantage is established.')
            decision={'selected_candidate_id':selected['candidate_id'],'recommendation':recommendation,
              'rationale':rationale,
              'tradeoffs':['Historical relationships do not establish present capacity, pricing, transit time, or willingness.',
                'Verify unresolved operational constraints before execution.'],
              'assumptions':[f"Scenario duration: {payload.get('scenario',{}).get('duration_days')} days; severity: {payload.get('scenario',{}).get('severity')} (user-supplied)."],
              'unresolved_checks':_as_text_list(selected.get('unresolved_checks')),'abstained':False}
        decision = _enforce_conditional_execution_boundary(decision, candidates, payload.get('scenario', {}).get('duration_days'))
        decision=DecisionOutput.model_validate(decision).model_dump()

    if GOOGLE_API_KEY:
        errors.extend(gemini_errors)
    else:
        errors.append('Gemini is disabled because GOOGLE_API_KEY is missing. Add a valid key to the project .env file and restart the server.')
    agent_errors=[]
    agents={
      'operations':_validate_agent(result.get('operations',{}),'Operations Agent',agent_errors),
      'risk':_validate_agent(result.get('risk',{}),'Risk Agent',agent_errors),
      'strategy':_validate_agent(result.get('strategy',{}),'Strategy Agent',agent_errors),
    }
    errors.extend(agent_errors)
    # Report the actual source mix, not a count of warnings. Previously, four fallback
    # reports could still be labelled 'Gemini + fallback' because the decision warning
    # was stored separately from the agent warnings.
    agent_sources=[a.get('source', 'deterministic_fallback') for a in agents.values()]
    any_gemini_used=any(source in {'gemini','mixed'} for source in agent_sources) or (decision_was_gemini if 'decision_was_gemini' in locals() else False)
    all_gemini_used=all(source == 'gemini' for source in agent_sources) and (decision_was_gemini if 'decision_was_gemini' in locals() else False) and not errors
    if not GOOGLE_API_KEY or not any_gemini_used:
        mode='deterministic_fallback'
    elif all_gemini_used:
        mode='gemini'
    else:
        mode='mixed_gemini_fallback'
    return {'mode':mode,'agents':agents,'decision':decision,'workflow_warnings':errors}
