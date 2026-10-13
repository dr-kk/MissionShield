"""E24-B: frozen E20-D cohort, restoration-strength and duration sensitivity.
Place in D:\\MissionShield\\scripts and run: python .\\scripts\\e24_restoration_sensitivity.py
No fitting, threshold changes, or outcome-based tuning. Original simulator functions reused.
"""
from pathlib import Path
import importlib.util, json
import numpy as np
import pandas as pd

ROOT=Path(r'D:\MissionShield')
SRC=ROOT/'results'/'e20d_fresh'
OUT=ROOT/'results'/'e24_restoration_sensitivity'
TARGET='edge_processing'
MULTIPLIERS=(0.45,0.70,0.90)
DURATIONS=('persistent',20)

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def replay(e0, e5b, attack, seed, t0, multiplier, duration):
    """Same RNG sequence and dynamics as E8, except specified restore multiplier/duration."""
    rng=np.random.default_rng(seed)
    onset,severity=e0.generate_run_profile(attack,rng)
    latent={n:float(rng.uniform(0.,0.03)) for n in e0.BASE_NODES}
    disturbance={'remaining':0,'strength':0.0}
    task_history=[]; timeline=[]
    for t in range(e0.STEPS_PER_RUN):
        level=e0.attack_intensity(attack,t,onset,severity)
        benign=e0.benign_disturbance(t,rng,disturbance)
        active=(t>=t0 and (duration=='persistent' or t<t0+int(duration)))
        # Preserve e5b.update_damage_cf exactly: alter only selected node's
        # previous-damage multiplier for restore_service, no other dynamics.
        previous=latent
        new={}
        attacked=e0.target_nodes(attack)
        for idx,node in enumerate(e0.BASE_NODES):
            direct=level if node in attacked else 0.0
            upstream=previous[e0.BASE_NODES[idx-1]] if idx>0 else 0.0
            previous_node=previous[node]*multiplier if (active and node==TARGET) else previous[node]
            damage=0.82*previous_node+0.12*direct+0.08*upstream
            damage+=0.035*benign+rng.normal(0.0,0.012)
            if direct<0.05: damage-=0.025
            new[node]=float(np.clip(damage,0.,1.))
        latent=new
        for node in e0.BASE_NODES:
            e0.generate_telemetry(node,attack,level,benign,latent[node],rng)
        p=e0.mission_task_probability(latent,rng)
        if active: p-=0.005
        p=float(np.clip(p,0.02,0.995))
        tasks=rng.binomial(1,p,size=8)
        task_history.extend(tasks.tolist())
        me=float(np.mean(task_history[-e0.MISSION_WINDOW*8:]))
        timeline.append((t,me,int(me<e0.MISSION_CRIT)))
    return int(onset),pd.DataFrame(timeline,columns=['time','mission_effectiveness','failure'])

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    e8=load(ROOT/'scripts'/'e8_independent_cohort_confirmation.py','e8_e24b')
    e0=e8.e0
    e5b=e8.e5b
    protocol=json.loads((SRC/'e20d_protocol.json').read_text())
    assert protocol['target']==TARGET and protocol['action']=='restore_service'
    actions=pd.read_csv(SRC/'e20d_actions.csv')
    assert len(actions)==2160
    assert actions.groupby(['model','horizon']).run_id.nunique().eq(180).all()
    family_map={f'e20d_{fi:02d}_{j:03d}':(family,int(protocol['seed_base'])+fi*10000+j)
                for fi,family in enumerate(e0.ATTACK_FAMILIES)
                for j in range(int(protocol['missions_per_family']))}
    assert len(family_map)==180 and set(actions.run_id)==set(family_map)
    selected=actions[actions.model=='MissionShield'].copy()
    assert set(selected.horizon)=={10,30,60}
    cache={}; rows=[]
    for r in selected.itertuples(index=False):
        rid=str(r.run_id); attack,seed=family_map[rid]
        for multiplier in MULTIPLIERS:
            for duration in DURATIONS:
                if int(r.alarm)==0:
                    pf=int(r.baseline_failure); gain=0.0
                else:
                    t0=int(r.alarm_time)
                    key=(rid,t0,multiplier,str(duration))
                    if key not in cache:
                        onset,cf=replay(e0,e5b,attack,seed,t0,multiplier,duration)
                        assert len(cf)==e0.STEPS_PER_RUN
                        cache[key]=cf
                    cf=cache[key]
                    pf=int(cf.failure.max())
                    gain=float(cf.mission_effectiveness.mean()) # raw post mean, not baseline-subtracted
                    if multiplier==0.45 and duration=='persistent':
                        # Exact original E8 emulator equivalence on one replay per alarm time.
                        check_key=(rid,t0)
                        if check_key not in cache:
                            original_onset,orig=e8.simulate_explicit_target(attack,seed,t0,TARGET)
                            assert int(original_onset)==onset
                            assert np.array_equal(orig.failure.to_numpy(),cf.failure.to_numpy()), f'Failure mismatch: {rid}'
                            assert np.allclose(orig.mission_effectiveness,cf.mission_effectiveness,rtol=0,atol=1e-12), f'RNG/dynamics mismatch: {rid}'
                            cache[check_key]=True
                        assert pf==int(r.post_failure), f'E20-D original mismatch: {rid} H{r.horizon}'
                rows.append({'run_id':rid,'horizon':int(r.horizon),'multiplier':multiplier,
                             'duration':str(duration),'alarm':int(r.alarm),
                             'baseline_failure':int(r.baseline_failure),'post_failure':pf,
                             'prevented':int(int(r.baseline_failure)==1 and pf==0),
                             'induced':int(int(r.baseline_failure)==0 and pf==1)})
    detail=pd.DataFrame(rows)
    detail.to_csv(OUT/'e24b_mission_outcomes.csv',index=False)
    summary=detail.groupby(['horizon','multiplier','duration'],as_index=False).agg(
        missions=('run_id','size'),actions=('alarm','sum'),baseline_failures=('baseline_failure','sum'),
        post_failures=('post_failure','sum'),prevented=('prevented','sum'),induced=('induced','sum'))
    summary['net_prevented']=summary.baseline_failures-summary.post_failures
    summary.to_csv(OUT/'e24b_summary.csv',index=False)
    (OUT/'e24b_protocol.json').write_text(json.dumps({
        'cohort':'frozen E20-D 180 missions','model':'MissionShield',
        'horizons':[10,30,60],'target':TARGET,'action':'restore_service',
        'multipliers':list(MULTIPLIERS),'duration_steps':['persistent',20],
        'cost_note':'Action costs are not recomputed; duration-dependent cost model not specified',
        'control':'0.45 persistent must reproduce original E20-D mission outcomes and E8 replay exactly',
        'exploratory':True,'no_model_retraining':True},indent=2))
    print('E24-B | RESTORATION-STRENGTH AND DURATION SENSITIVITY')
    print(summary.to_string(index=False))
    print('[PASS] 0.45 persistent original replay verified for every distinct alarm time')
    print('Saved:',OUT)

if __name__=='__main__': main()
