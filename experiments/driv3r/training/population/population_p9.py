"""P9: evolutionary game-policy versus scenario-generator curriculum tournament.

Synthetic, externally generated host-legal grid games. The adversary proposes
obstacle/protected/goal configurations. Agents are real small MLPs trained with
Adam imitation of a handcrafted action utility (NOT game RL, true self-play,
SLM training, native CyberSand, or learned aesthetic preference).

A controlled tournament compares uniform scenario choice, fixed pre-training
hardness selection, and on-policy adaptive hardness. Every arm evaluates the
same number of candidates, with explicit probe/train accounting; independent
train/selection/heldout seeds and untrained scenario variants isolate claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .population_lab import NICHES, DIRECTIONS, World, softmax, utility, stable_seed, draw_svg
    from .population_p1 import load_policy, model_sha
except ImportError:
    from population_lab import NICHES, DIRECTIONS, World, softmax, utility, stable_seed, draw_svg
    from population_p1 import load_policy, model_sha

ARMS = ("random_scenarios", "frozen_hardness", "adaptive_hardness")
CARDS = ("base", "dead_end", "slalom", "protected_choke", "near_goal_block", "goal_flip",
         "crosswall_surprise", "goal_lure_surprise")
TRAIN_CARD_COUNT = 6


def scenario_world(seed, card, horizon=32):
    """Deterministic synthetic material/obstacle worlds, preserving legal start."""
    if not 0 <= card < len(CARDS) or not 16 <= horizon <= 128:
        raise ValueError("invalid synthetic scenario")
    w = World.generate(seed, max_steps=horizon)
    if card == 1:
        # Deterministic one-cell painting cul-de-sac: RETRACE is available.
        w.obstacles.update({(5,11), (4,10), (4,12)})
        w.obstacles.discard((4,11))
    elif card == 2:
        w.obstacles.update((x, 8 if x % 2 else 14) for x in range(5,19))
        w.obstacles.update((8,y) for y in range(4,19) if y not in (7,13,17))
    elif card == 3:
        w.protected.update((x,y) for x in range(12,16) for y in range(7,16) if (x+y+seed)%4!=0)
        w.goal = (18,4)
        w.second_goal = (4,18)
    elif card == 4:
        w.obstacles.update((x,y) for x in range(15,20) for y in range(8,15) if x != 19 or y != 11)
        w.goal = (19,11)
        w.second_goal = (18,18)
    elif card == 5:
        w.goal = (3,4)
        w.second_goal = (19,17)
        w.obstacles.update((x,7) for x in range(2,15) if x not in (5,12))
    elif card == 6:
        # Out-of-training-family: two staggered long walls and narrow gateways.
        w.obstacles.update((x,6) for x in range(2,20) if x not in (4,14))
        w.obstacles.update((x,16) for x in range(2,20) if x not in (9,18))
        w.goal=(19,4); w.second_goal=(4,18)
    elif card == 7:
        # Out-of-training-family: lure the agent toward a largely blocked goal.
        w.goal=(18,4); w.second_goal=(4,17)
        w.protected.update((x,y) for x in range(15,20) for y in range(2,8)
                           if not (x==18 and y==4))
        w.obstacles.update((x,y) for x in range(12,16) for y in (3,7))
    w.obstacles.difference_update(w.protected | w.painted | {(3,11), w.goal, w.second_goal})
    w.protected.difference_update(w.painted | {(3,11),w.goal,w.second_goal})
    if w.painted & (w.obstacles | w.protected):
        raise AssertionError("invalid initial material map")
    return w


def play(policy, niche, seed, card, horizon=32):
    w = scenario_world(seed, card, horizon)
    reward = 0.
    while not w.stopped and w.steps < w.max_steps:
        menu = w.legal()
        X = np.stack([w.feature(a) for a in menu])
        if policy == "oracle":
            chosen = int(np.argmax([utility(niche, f, w) for f in X]))
        elif policy == "nonstop_random":
            valid = [i for i,a in enumerate(menu) if a != "STOP"]
            r = np.random.default_rng(stable_seed("p9",seed,niche,card,w.steps,"random"))
            chosen = int(r.choice(valid)) if valid else menu.index("STOP")
        else:
            chosen = policy.choice(X)
        reward += utility(niche, X[chosen], w)
        w.step(menu[chosen])
        if w.painted & (w.protected | w.obstacles):
            raise AssertionError("illegal cell material mutation")
    return dict(reward=float(reward), painted=len(w.painted)-1,
                retraces=sum(h["action"]=="RETRACE" for h in w.history),
                early_stop=int(w.stopped and w.steps < horizon),
                goal_switch=int(w.switched),
                goal_distance=abs(w.cursor[0]-w.goal[0])+abs(w.cursor[1]-w.goal[1]),
                legal_actions=len(w.history),
                world=w)


def hardness(policy,niche,seed,card,horizon=32, oracle_reward=None):
    """TRAIN-only challenge score: normalized greedy-oracle regret + STOP penalty."""
    if oracle_reward is None:
        oracle_reward=play("oracle",niche,seed,card,horizon)["reward"]
    trial=play(policy,niche,seed,card,horizon)
    return float(np.clip((oracle_reward-trial["reward"])/(1+abs(oracle_reward)), -2, 2)
                 + .35*trial["early_stop"])


def train_episode(policy,niche,seed,card,horizon=32):
    """Actual teacher-augmented on-policy Adam; same optimizer/action budget rules."""
    w=scenario_world(seed,card,horizon)
    rng=np.random.default_rng(stable_seed("p9",seed,niche,"rollout"))
    losses=[]
    before=policy.updates
    while not w.stopped and w.steps < w.max_steps:
        menu=w.legal()
        X=np.stack([w.feature(a) for a in menu])
        targets=softmax(np.asarray([utility(niche,x,w) for x in X])/.42)
        losses.append(policy.teach(X,targets))
        chosen=int(rng.choice(len(menu),p=targets)) if rng.random()<.25 else policy.choice(X,rng)
        w.step(menu[chosen])
    if w.painted & (w.protected | w.obstacles) or policy.updates <= before:
        raise AssertionError("invalid training episode")
    return dict(steps=policy.updates-before,loss=float(np.mean(losses)),
                decisions=len(w.history),retraces=sum(h["action"]=="RETRACE" for h in w.history))


def split_seeds(seed, generations, episodes, selection, holdout):
    out=dict(train=[[stable_seed("p9",seed,"train",g,i) for i in range(episodes)]
                    for g in range(generations)],
             selection=[stable_seed("p9",seed,"selection",i) for i in range(selection)],
             holdout=[stable_seed("p9",seed,"holdout",i) for i in range(holdout)])
    flat=sum(out["train"],[])+out["selection"]+out["holdout"]
    if len(flat)!=len(set(flat)):
        raise AssertionError("training/selection/holdout seed leakage")
    return out


def offered_cards(seed,niche,g,i,episode_index,n=3):
    """Identical fixed TRAIN-only scenario proposals for all arms."""
    r=np.random.default_rng(stable_seed("p9",seed,niche,g,i,episode_index,"cards"))
    return [int(x) for x in r.choice(TRAIN_CARD_COUNT,size=n,replace=False)]


def choose_card(arm, policy, frozen_source, niche, scenario_seed, offered, random_index,
                horizon, oracle_cache):
    """Same probe budget for all arms. Frozen-source probes never update."""
    live=[]; frozen=[]
    for card in offered:
        key=(niche,scenario_seed,card,horizon)
        if key not in oracle_cache:
            oracle_cache[key]=play("oracle",niche,scenario_seed,card,horizon)["reward"]
        target=oracle_cache[key]
        live.append(hardness(policy,niche,scenario_seed,card,horizon,target))
        frozen.append(hardness(frozen_source,niche,scenario_seed,card,horizon,target))
    if arm=="random_scenarios":
        index=random_index%len(offered)
    elif arm=="frozen_hardness":
        index=int(np.argmax(frozen))
    elif arm=="adaptive_hardness":
        index=int(np.argmax(live))
    else:
        raise ValueError("invalid adversary")
    return offered[index], dict(offered=offered, live=np.round(live,5).tolist(),
                                 frozen=np.round(frozen,5).tolist(),selected_index=index)


def score_policy(policy,niche,seeds,cards,horizon):
    worlds=[play(policy,niche,s,c,horizon) for s in seeds for c in cards]
    return dict(mean_reward=round(float(np.mean([w["reward"] for w in worlds])),6),
                mean_painted=round(float(np.mean([w["painted"] for w in worlds])),5),
                mean_retraces=round(float(np.mean([w["retraces"] for w in worlds])),5),
                early_stop_rate=round(float(np.mean([w["early_stop"] for w in worlds])),5),
                goal_switch_rate=round(float(np.mean([w["goal_switch"] for w in worlds])),5),
                mean_goal_distance=round(float(np.mean([w["goal_distance"] for w in worlds])),5),
                mean_descriptor=np.mean([w["world"].descriptor() for w in worlds],axis=0).round(6).tolist(),
                episodes=len(worlds),decisions=sum(w["legal_actions"] for w in worlds))


def run_p9(p0_output,output,seed=640,generations=2,offspring=3,train_episodes=7,
           selection_episodes=4,holdout_episodes=6,horizon=40,portfolio=2,proposals=3):
    if not (2<=generations<=4 and 2<=offspring<=8 and 4<=train_episodes<=20 and
            3<=selection_episodes<=20 and 4<=holdout_episodes<=24 and
            32<=horizon<=96 and 1<=portfolio<=offspring and 2<=proposals<=5):
        raise ValueError("unbounded campaign")
    p0_output,output=Path(p0_output),Path(output)
    provenance=json.loads((p0_output/"results.json").read_text())
    if provenance.get("format")!="population-research-v1" or provenance.get("language_model") is not False:
        raise ValueError("P9 requires source-pinned P0 synthetic policy checkpoint")
    splits=split_seeds(seed,generations,train_episodes,selection_episodes,holdout_episodes)
    p0_seeds={stable_seed(provenance["seed"],name,j) for name,count in
              (("train",provenance["train_seed_count"]),
               ("selection",provenance["selection_seed_count"]),
               ("test",provenance["heldout_seed_count"])) for j in range(count)}
    all_seeds=sum(splits["train"],[])+splits["selection"]+splits["holdout"]
    if set(all_seeds)&p0_seeds:
        raise AssertionError("P0/P9 seed overlap")
    parents={n:load_policy(p0_output/"models"/(n+".json")) for n in NICHES}
    output.mkdir(parents=True,exist_ok=False)
    (output/"models").mkdir()
    (output/"gallery").mkdir()
    models,records,comparisons={},[],{}
    oracle_cache={}
    for niche in NICHES:
        comparisons[niche]={}
        frozen=parents[niche]
        for arm in ARMS:
            candidates=[]
            parent=frozen
            for g in range(generations):
                for i in range(offspring):
                    mutation_seed=stable_seed("p9",seed,niche,g,i,"mutation")
                    policy=parent.mutate(mutation_seed)
                    ident=f"{niche}-{arm}-g{g}-{i}"
                    initial=model_sha(policy)
                    choices=[];updates=0;decisions=0;retrace=0;losses=[]
                    for ep, world_seed in enumerate(splits["train"][g]):
                        # Proposals depend on niche/generation/offspring/episode, NEVER arm.
                        cards=offered_cards(seed,niche,g,i,ep,proposals)
                        random_index=stable_seed("p9",seed,niche,g,i,ep,"random")
                        card,probe=choose_card(arm,policy,frozen,niche,world_seed,cards,
                                               random_index,32,oracle_cache)
                        t=train_episode(policy,niche,world_seed,card,32)
                        choices.append(dict(episode=ep,seed=world_seed,card=card,
                                            card_name=CARDS[card],probe=probe))
                        updates+=t["steps"]
                        decisions+=t["decisions"]
                        retrace+=t["retraces"]
                        losses.append(t["loss"])
                    selected=score_policy(policy,niche,splits["selection"],range(TRAIN_CARD_COUNT),32)
                    rec=dict(id=ident,niche=niche,arm=arm,generation=g,offspring=i,
                             parent=None if g==0 else parent_name,mutation_seed=mutation_seed,
                             initial_sha256=initial,final_sha256=model_sha(policy),
                             lr=policy.lr,temperature=policy.temperature,train_cards=choices,
                             training=dict(adam_steps=updates,curriculum_decisions=decisions,
                                           retraces=retrace,probe_policy_rollouts=len(choices)*proposals*2,
                                           probe_oracle_rollouts=len(choices)*proposals,
                                           mean_loss=round(float(np.mean(losses)),6)),selection=selected)
                    if rec["initial_sha256"]==rec["final_sha256"]:
                        raise AssertionError("failed actual optimizer update")
                    records.append(rec);candidates.append(rec);models[ident]=policy
                    (output/"models"/(ident+".json")).write_text(json.dumps(policy.serialized(),sort_keys=True)+"\n")
                # Selection-only parent chosen from this generation, no test access.
                best=max((r for r in candidates if r["generation"]==g),
                         key=lambda r:(r["selection"]["mean_reward"],r["id"]))
                parent=models[best["id"]]
                parent_name=best["id"]
            ranked=sorted(candidates,key=lambda r:(-r["selection"]["mean_reward"],r["id"]))
            chosen=ranked[:portfolio]
            comp=dict(champion_id=ranked[0]["id"],selected_ids=[r["id"] for r in chosen],
                      selection_reward=ranked[0]["selection"]["mean_reward"],
                      selected_generation_counts={str(g):sum(r["generation"]==g for r in chosen)
                                                  for g in range(generations)},
                      train_card_histogram={CARDS[k]:sum(v["card"]==k for r in candidates
                                                             for v in r["train_cards"])
                                            for k in range(TRAIN_CARD_COUNT)})
            # Strict final holdout gate: no held-out world used in training or parent selection.
            for family,cards in (("familiar",range(TRAIN_CARD_COUNT)),
                                 ("unseen_geometry",range(TRAIN_CARD_COUNT,len(CARDS)))):
                metrics=[score_policy(models[r["id"]],niche,splits["holdout"],cards,horizon)
                         for r in chosen]
                pairs=[float(np.linalg.norm(np.asarray(x["mean_descriptor"])-
                           np.asarray(y["mean_descriptor"])))
                       for j,x in enumerate(metrics) for y in metrics[j+1:]]
                comp[family]=dict(
                    mean_reward=round(float(np.mean([x["mean_reward"] for x in metrics])),6),
                    early_stop_rate=round(float(np.mean([x["early_stop_rate"] for x in metrics])),6),
                    mean_retraces=round(float(np.mean([x["mean_retraces"] for x in metrics])),6),
                    mean_painted=round(float(np.mean([x["mean_painted"] for x in metrics])),6),
                    goal_switch_rate=round(float(np.mean([x["goal_switch_rate"] for x in metrics])),6),
                    diversity=round(float(np.mean(pairs)) if pairs else 0.,6),
                    per_member=metrics)
            comparisons[niche][arm]=comp
            for j,r in enumerate(chosen):
                w=play(models[r["id"]],niche,splits["holdout"][0],7,horizon)["world"]
                path=f"{niche}-{arm}-{j}"
                (output/"gallery"/(path+".svg")).write_text(draw_svg(w))
                (output/"gallery"/(path+"-trace.json")).write_text(json.dumps(dict(
                    synthetic=True,native=False,slm=False,seed=splits["holdout"][0],
                    card=CARDS[7],policy_id=r["id"],trajectory=w.history,painted=sorted(w.painted),
                    obstacles=sorted(w.obstacles),protected=sorted(w.protected)),indent=2)+"\n")
    summary={arm:{family:{metric:round(float(np.mean([comparisons[n][arm][family][metric]
                                                       for n in NICHES])),6)
                          for metric in ("mean_reward","early_stop_rate","mean_retraces",
                                         "mean_painted","diversity","goal_switch_rate")}
                  for family in ("familiar","unseen_geometry")} for arm in ARMS}
    result=dict(format="population-research-p9",kind="SYNTHETIC_SCENARIO_ADVERSARY_CURRICULUM",
                synthetic=True,native=False,language_model=False,reinforcement_learning=False,
                actual_self_play=False,game_code_modified=False,seed=seed,arms=list(ARMS),
                card_names=list(CARDS),train_card_count=TRAIN_CARD_COUNT,
                generations=generations,offspring=offspring,portfolio=portfolio,horizon=horizon,
                candidates=len(records),adam_updates=sum(x["training"]["adam_steps"] for x in records),
                train_decisions=sum(x["training"]["curriculum_decisions"] for x in records),
                matched_probe_budget=sum(x["training"]["probe_policy_rollouts"] for x in records),
                lineage=records,comparison=comparisons,summary=summary,
                training_seed_hash=hashlib.sha256(json.dumps(splits["train"]).encode()).hexdigest(),
                selection_seed_hash=hashlib.sha256(json.dumps(splits["selection"]).encode()).hexdigest(),
                heldout_seed_hash=hashlib.sha256(json.dumps(splits["holdout"]).encode()).hexdigest(),
                limitations=["External synthetic worlds, not native CyberSand or Workbench",
                  "Actual supervised Adam neural policies, NOT SLM adapters or game RL",
                  "Adversary selects scenario cards, not learned opponent actions or true self-play",
                  "World families 6/7 are unseen card templates but reuse the same base generator",
                  "All policy decisions host-masked legal, not learned illegal-action avoidance",
                  "Hardness score uses handcrafted per-niche utility, not human aesthetic judgement",
                  "Fixed and adaptive challenger arms use matching probe counts but separate training state distributions",
                  "Only selection-frozen portfolios are evaluated on heldouts; small cohort counts limit generality"])
    (output/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--p0-output",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seed",type=int,default=640)
    p.add_argument("--generations",type=int,default=2)
    p.add_argument("--offspring",type=int,default=3)
    p.add_argument("--train-episodes",type=int,default=7)
    p.add_argument("--selection-episodes",type=int,default=4)
    p.add_argument("--holdout-episodes",type=int,default=6)
    p.add_argument("--horizon",type=int,default=40)
    p.add_argument("--portfolio",type=int,default=2)
    p.add_argument("--proposals",type=int,default=3)
    a=p.parse_args()
    r=run_p9(a.p0_output,a.output,a.seed,a.generations,a.offspring,
             a.train_episodes,a.selection_episodes,a.holdout_episodes,a.horizon,
             a.portfolio,a.proposals)
    print(json.dumps(dict(seed=r["seed"],candidates=r["candidates"],
                          adam_updates=r["adam_updates"],summary=r["summary"]),indent=2))

if __name__=="__main__":
    main()
