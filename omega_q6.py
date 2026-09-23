
import math, time, argparse
import numpy as np

class DSU:
    __slots__ = ("p","sz")
    def __init__(self,n):
        self.p=list(range(n)); self.sz=[1]*n
    def find(self,x):
        while self.p[x]!=x:
            self.p[x]=self.p[self.p[x]]
            x=self.p[x]
        return x
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a==b: return False
        if self.sz[a]<self.sz[b]: a,b=b,a
        self.p[b]=a; self.sz[a]+=self.sz[b]
        return True

def tour_length(D,tour):
    n=len(tour)
    return float(sum(D[tour[i],tour[(i+1)%n]] for i in range(n)))

def _cycle_from_edges(n,edges):
    adj=[[] for _ in range(n)]
    for u,v in edges:
        adj[u].append(v); adj[v].append(u)
    if any(len(x)!=2 for x in adj):
        raise RuntimeError("not 2-regular")
    tour=[0]; prev=-1; cur=0
    for _ in range(n-1):
        a,b=adj[cur]
        nxt=a if a!=prev else b
        tour.append(nxt); prev,cur=cur,nxt
    if tour[-1] not in adj[0]:
        raise RuntimeError("multiple cycles")
    return tour

def _one_full_two_opt_pass(D,tour):
    n=len(tour); best_delta=0.0; best=None
    for i in range(n-1):
        a=tour[i]; b=tour[(i+1)%n]
        for j in range(i+2, n if i>0 else n-1):
            c=tour[j]; d=tour[(j+1)%n]
            delta=D[a,c]+D[b,d]-D[a,b]-D[c,d]
            if delta<best_delta:
                best_delta=float(delta); best=(i+1,j)
    if best:
        l,r=best
        tour=tour[:l]+tour[l:r+1][::-1]+tour[r+1:]
    return tour

def omega_q2(D,beta=0.5,two_opt_pass=True):
    D=np.asarray(D,dtype=np.float64)
    n=D.shape[0]
    if D.shape!=(n,n) or n<3: raise ValueError("D must be n x n, n>=3")
    W=D.copy(); np.fill_diagonal(W,np.inf)
    two=np.argpartition(W,kth=1,axis=1)[:,:2]
    vals=np.take_along_axis(W,two,axis=1)
    ord2=np.argsort(vals,axis=1)
    first_idx=two[np.arange(n),ord2[:,0]]
    first_val=vals[np.arange(n),ord2[:,0]]
    second_val=vals[np.arange(n),ord2[:,1]]

    iu,iv=np.triu_indices(n,1)
    w=D[iu,iv]
    alt_u=np.where(first_idx[iu]==iv,second_val[iu],first_val[iu])
    alt_v=np.where(first_idx[iv]==iu,second_val[iv],first_val[iv])
    score=w-beta*((alt_u-w)+(alt_v-w))
    order=np.argsort(score,kind="stable")

    deg=np.zeros(n,dtype=np.int8)
    dsu=DSU(n); edges=[]
    for idx in order:
        u=int(iu[idx]); v=int(iv[idx])
        if deg[u]>=2 or deg[v]>=2: continue
        if dsu.find(u)==dsu.find(v): continue
        edges.append((u,v)); deg[u]+=1; deg[v]+=1; dsu.union(u,v)
        if len(edges)==n-1: break
    if len(edges)!=n-1: raise RuntimeError("failed to form Hamiltonian path")
    ends=np.flatnonzero(deg==1)
    edges.append((int(ends[0]),int(ends[1])))
    tour=_cycle_from_edges(n,edges)
    if two_opt_pass:
        tour=_one_full_two_opt_pass(D,tour)
    return tour

def held_karp_exact(D):
    D=np.asarray(D,dtype=np.float64); n=D.shape[0]; m=n-1; size=1<<m
    inf=float("inf")
    dp=np.full((size,m),inf); parent=np.full((size,m),-1,dtype=np.int16)
    for j in range(m): dp[1<<j,j]=D[0,j+1]
    for mask in range(1,size):
        bits=mask
        while bits:
            lb=bits&-bits; j=lb.bit_length()-1; pm=mask^lb
            if pm:
                best=inf; bestk=-1; kb=pm
                while kb:
                    klb=kb&-kb; k=klb.bit_length()-1
                    val=dp[pm,k]+D[k+1,j+1]
                    if val<best: best=val; bestk=k
                    kb^=klb
                dp[mask,j]=best; parent[mask,j]=bestk
            bits^=lb
    full=size-1; best=inf; last=-1
    for j in range(m):
        val=dp[full,j]+D[j+1,0]
        if val<best: best=val; last=j
    rev=[]; mask=full; j=last
    while j>=0:
        rev.append(j+1); pj=int(parent[mask,j]); mask^=1<<j; j=pj
    return [0]+rev[::-1],float(best)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=12)
    ap.add_argument("--seed",type=int,default=20260922)
    args=ap.parse_args()
    rng=np.random.default_rng(args.seed)
    pts=rng.random((args.n,2))
    delta=pts[:,None,:]-pts[None,:,:]
    D=np.sqrt((delta*delta).sum(axis=2))
    t0=time.perf_counter(); tour=omega_q2(D); t1=time.perf_counter()
    print("OMEGA-Q2",tour_length(D,tour),"time_ms",(t1-t0)*1000)
    if args.n<=18:
        t0=time.perf_counter(); ot,oc=held_karp_exact(D); t1=time.perf_counter()
        print("EXACT",oc,"time_ms",(t1-t0)*1000)
        print("gap_%",(tour_length(D,tour)/oc-1)*100)


def _two_opt_once(D,tour):
    n=len(tour); best_delta=0.0; pair=None
    for i in range(n-1):
        a=tour[i]; b=tour[(i+1)%n]
        for j in range(i+2,n if i>0 else n-1):
            c=tour[j]; d=tour[(j+1)%n]
            delta=D[a,c]+D[b,d]-D[a,b]-D[c,d]
            if delta<best_delta:
                best_delta=float(delta); pair=(i+1,j)
    if pair is not None:
        l,r=pair
        return tour[:l]+tour[l:r+1][::-1]+tour[r+1:]
    return tour

def omega_q2_plus(D,betas=(0.0,0.25,0.5,1.0,2.0),two_opt_passes=4):
    best=None; best_cost=float("inf")
    for beta in betas:
        t=omega_q2(D,beta=beta,two_opt_pass=False)
        c=tour_length(D,t)
        for _ in range(two_opt_passes):
            nt=_two_opt_once(D,t)
            nc=tour_length(D,nt)
            if nc>=c-1e-12: break
            t,c=nt,nc
        if c<best_cost:
            best,best_cost=t,c
    return best


# ============================================================
# OMEGA-Q2 STRICT PORTFOLIO
# All auxiliary search passes below are O(n^2).
# Current overall bound: O(n^2 log n), because omega_q2()
# comparison-sorts Theta(n^2) edge scores.
# ============================================================

def _normalize_tour0(t):
    k=t.index(0)
    return t[k:]+t[:k]

def _two_opt_once_q2(D,tour):
    n=len(tour); best_delta=0.0; best=None
    for i in range(n-1):
        a=tour[i]; b=tour[(i+1)%n]
        for j in range(i+2, n if i>0 else n-1):
            c=tour[j]; d=tour[(j+1)%n]
            delta=D[a,c]+D[b,d]-D[a,b]-D[c,d]
            if delta<best_delta-1e-15:
                best_delta=float(delta); best=(i+1,j)
    if best is None: return tour,False
    l,r=best
    return tour[:l]+tour[l:r+1][::-1]+tour[r+1:],True

def _relocate_once_q2(D,tour):
    n=len(tour); best_delta=0.0; best=None
    for i in range(n):
        p=tour[(i-1)%n]; x=tour[i]; q=tour[(i+1)%n]
        removal=D[p,q]-D[p,x]-D[x,q]
        for j in range(n):
            if j==i or j==(i-1)%n: continue
            a=tour[j]; b=tour[(j+1)%n]
            if b==x: continue
            delta=removal+D[a,x]+D[x,b]-D[a,b]
            if delta<best_delta-1e-15:
                best_delta=float(delta); best=(i,j)
    if best is None: return tour,False
    i,j=best; x=tour[i]; t=tour[:]; t.pop(i)
    if i<j: j-=1
    t.insert(j+1,x)
    return t,True

def _swap_once_q2(D,tour):
    n=len(tour); best_delta=0.0; best=None
    def after(k,i,j):
        if k==i: return tour[j]
        if k==j: return tour[i]
        return tour[k]
    for i in range(1,n):
        for j in range(i+1,n):
            affected={(i-1)%n,i,(j-1)%n,j}
            old=0.0; new=0.0
            for e in affected:
                f=(e+1)%n
                old+=D[tour[e],tour[f]]
                new+=D[after(e,i,j),after(f,i,j)]
            delta=float(new-old)
            if delta<best_delta-1e-15:
                best_delta=delta; best=(i,j)
    if best is None: return tour,False
    t=tour[:]; i,j=best; t[i],t[j]=t[j],t[i]
    return t,True

def _polish_q2(D,tour,rounds=8):
    t=tour[:]
    for _ in range(rounds):
        changed=False
        t,ch=_two_opt_once_q2(D,t); changed|=ch
        t,ch=_relocate_once_q2(D,t); changed|=ch
        t,ch=_swap_once_q2(D,t); changed|=ch
        if not changed: break
    return t

def _nearest_neighbor_start_q2(D,start):
    n=len(D); used=np.zeros(n,dtype=bool); used[start]=True
    t=[int(start)]; cur=int(start)
    for _ in range(n-1):
        row=D[cur].copy(); row[used]=np.inf
        nxt=int(np.argmin(row))
        t.append(nxt); used[nxt]=True; cur=nxt
    return _normalize_tour0(t)

def _insertion_q2(D,seed=0,mode="farthest"):
    n=len(D)
    unused=np.ones(n,dtype=bool); unused[seed]=False
    if mode=="farthest":
        partner=int(np.argmax(np.where(unused,D[seed],-np.inf)))
    else:
        partner=int(np.argmin(np.where(unused,D[seed],np.inf)))
    tour=[seed,partner]; unused[partner]=False
    mind=np.minimum(D[:,seed],D[:,partner])
    mind[~unused]=np.inf if mode=="nearest" else -np.inf
    while len(tour)<n:
        if mode=="farthest":
            x=int(np.argmax(np.where(unused,mind,-np.inf)))
        else:
            x=int(np.argmin(np.where(unused,mind,np.inf)))
        best_inc=float("inf"); best_pos=1
        m=len(tour)
        for i in range(m):
            a=tour[i]; b=tour[(i+1)%m]
            inc=D[a,x]+D[x,b]-D[a,b]
            if inc<best_inc:
                best_inc=float(inc); best_pos=i+1
        tour.insert(best_pos,x); unused[x]=False
        for u in range(n):
            if unused[u] and D[u,x]<mind[u]:
                mind[u]=D[u,x]
        mind[x]=np.inf if mode=="nearest" else -np.inf
    return _normalize_tour0(tour)

def _fixed_kicks_q2(tour):
    n=len(tour)
    if n<8: return []
    cutsets=[]
    for shift in (0,1,2):
        a=max(1,n//5+shift)
        b=max(a+1,2*n//5+shift)
        c=max(b+1,3*n//5+shift)
        d=max(c+1,4*n//5+shift)
        if d<n: cutsets.append((a,b,c,d))
    extra=[
        (1,max(2,n//3),max(3,2*n//3),n-1),
        (max(1,n//4),max(2,n//2-1),max(3,3*n//4),n-1),
        (2,max(3,n//3+1),max(4,2*n//3+1),n-1),
    ]
    for a,b,c,d in extra:
        if 0<a<b<c<d<n: cutsets.append((a,b,c,d))
    out=[]
    for a,b,c,d in cutsets:
        A=tour[:a]; B=tour[a:b]; C=tour[b:c]; D=tour[c:d]; E=tour[d:]
        variants=[
            A+C+B+D+E,
            A+B+D+C+E,
            A+D+C+B+E,
            A+C+D+B+E,
            A+D+B+C+E,
            A+B+C[::-1]+D+E,
            A+B[::-1]+C+D+E,
            A+B+D[::-1]+C+E,
        ]
        for p in variants:
            if len(set(p))==n: out.append(_normalize_tour0(p))
    return out

def omega_q2_strict(D):
    """
    Experimental symmetric-TSP solver.

    Empirical status in the bundled Sep-2026 tests:
    exact optimum on all tested instances through n=16.

    This is NOT an exactness proof.

    Current comparison-model complexity:
        O(n^2 log n)
    from sorting Theta(n^2) edge scores in omega_q2().
    All portfolio counts and local-search rounds are fixed constants,
    and every auxiliary pass is O(n^2).
    """
    D=np.asarray(D,dtype=np.float64)
    n=len(D)
    candidates=[]
    for beta in (-1.0,-0.5,0.0,0.125,0.25,0.5,0.75,1.0,1.5,2.0,3.0,4.0):
        candidates.append(omega_q2(D,beta=beta,two_opt_pass=False))

    starts=list(range(min(n,16)))
    for s in starts:
        candidates.append(_nearest_neighbor_start_q2(D,s))
    for s in starts[:8]:
        candidates.append(_insertion_q2(D,s,"farthest"))
        candidates.append(_insertion_q2(D,s,"nearest"))

    best=None; best_cost=float("inf")
    for t in candidates:
        t=_polish_q2(D,t,rounds=8)
        c=tour_length(D,t)
        if c<best_cost:
            best,best_cost=t,c

    for kt in _fixed_kicks_q2(best)[:48]:
        t=_polish_q2(D,kt,rounds=10)
        c=tour_length(D,t)
        if c<best_cost:
            best,best_cost=t,c

    snapshot=best
    for kt in _fixed_kicks_q2(snapshot)[:24]:
        t=_polish_q2(D,kt,rounds=10)
        c=tour_length(D,t)
        if c<best_cost:
            best,best_cost=t,c
    return best


# ============================================================
# OMEGA-Q3
# Plateau-aware extension of OMEGA-Q2 STRICT.
#
# New idea:
#   permit a fixed-depth, fixed-width walk across zero-cost plateaus.
#   A chain of neutral moves is accepted only if it exposes a strict
#   improvement afterwards.
#
# Asymptotics:
#   Q2 core: O(n^2 log n) due to sorting Theta(n^2) edge scores.
#   Q3 bridge: depth, width, rounds are fixed constants; every scan
#              considers O(n^2) 2-opt/relocate/swap moves.
#   Therefore Q3 remains O(n^2 log n) in the comparison model.
# ============================================================

def _q3_normalize(t):
    t=list(t)
    k=t.index(0)
    t=t[k:]+t[:k]
    rev=[t[0]]+list(reversed(t[1:]))
    return tuple(t) if tuple(t)<=tuple(rev) else tuple(rev)

def _q3_apply_move(tour,mv):
    kind,a,b=mv
    t=list(tour)
    if kind=="2opt":
        return t[:a]+t[a:b+1][::-1]+t[b+1:]
    if kind=="relocate":
        x=t.pop(a)
        j=b
        if a<j:
            j-=1
        t.insert(j+1,x)
        return t
    if kind=="swap":
        t[a],t[b]=t[b],t[a]
        return t
    raise ValueError("unknown Q3 move")

def _q3_scan_moves(D,tour,neutral=False,neutral_cap=96):
    n=len(tour)
    best_delta=0.0
    best_move=None
    neutrals=[]

    # 2-opt
    for i in range(n-1):
        a,b=tour[i],tour[(i+1)%n]
        for j in range(i+2,n if i>0 else n-1):
            c,d=tour[j],tour[(j+1)%n]
            delta=float(D[a,c]+D[b,d]-D[a,b]-D[c,d])
            if delta<best_delta-1e-12:
                best_delta=delta
                best_move=("2opt",i+1,j)
            elif neutral and abs(delta)<=1e-12 and len(neutrals)<neutral_cap:
                neutrals.append(("2opt",i+1,j))

    # relocate
    for i in range(n):
        p=tour[(i-1)%n]
        x=tour[i]
        q=tour[(i+1)%n]
        removal=D[p,q]-D[p,x]-D[x,q]
        for j in range(n):
            if j==i or j==(i-1)%n:
                continue
            a=tour[j]
            b=tour[(j+1)%n]
            if b==x:
                continue
            delta=float(removal+D[a,x]+D[x,b]-D[a,b])
            if delta<best_delta-1e-12:
                best_delta=delta
                best_move=("relocate",i,j)
            elif neutral and abs(delta)<=1e-12 and len(neutrals)<neutral_cap:
                neutrals.append(("relocate",i,j))

    # swap with O(1) delta per pair
    def swap_delta(i,j):
        affected={(i-1)%n,i,(j-1)%n,j}
        def after(k):
            if k==i:
                return tour[j]
            if k==j:
                return tour[i]
            return tour[k]
        old=new=0.0
        for e in affected:
            f=(e+1)%n
            old+=D[tour[e],tour[f]]
            new+=D[after(e),after(f)]
        return float(new-old)

    for i in range(1,n):
        for j in range(i+1,n):
            delta=swap_delta(i,j)
            if delta<best_delta-1e-12:
                best_delta=delta
                best_move=("swap",i,j)
            elif neutral and abs(delta)<=1e-12 and len(neutrals)<neutral_cap:
                neutrals.append(("swap",i,j))

    return best_delta,best_move,neutrals

def _q3_strict_polish(D,tour,rounds=10):
    t=list(tour)
    for _ in range(rounds):
        _,mv,_=_q3_scan_moves(D,t,neutral=False)
        if mv is None:
            break
        t=_q3_apply_move(t,mv)
    return t

def _q3_plateau_bridge(D,tour,depth=2,beam_width=64,neutral_cap=96):
    """
    Search a constant-depth neutral plateau.

    A state is expanded only through zero-cost 2-opt/relocate/swap moves.
    As soon as a strict improvement becomes available, take it and repolish.

    Fixed depth and width keep the bridge O(n^2).
    """
    base=list(tour)
    base_cost=tour_length(D,base)

    beam=[(base,[])]
    seen={_q3_normalize(base)}

    for level in range(depth+1):
        next_states=[]

        for state,path in beam:
            _,mv,neutrals=_q3_scan_moves(
                D,state,
                neutral=(level<depth),
                neutral_cap=neutral_cap
            )

            if mv is not None:
                improved=_q3_apply_move(state,mv)
                improved=_q3_strict_polish(D,improved,rounds=10)
                if tour_length(D,improved)<base_cost-1e-12:
                    return improved,path+[mv]

            if level<depth:
                for nm in neutrals:
                    nt=_q3_apply_move(state,nm)
                    key=_q3_normalize(nt)
                    if key not in seen:
                        seen.add(key)
                        next_states.append((nt,path+[nm]))
                        if len(next_states)>=beam_width:
                            break
                if len(next_states)>=beam_width:
                    break

        if not next_states:
            break
        beam=next_states[:beam_width]

    return None,None

def omega_q3(D):
    """
    OMEGA-Q3 = unchanged OMEGA-Q2 STRICT + plateau bridge.

    Current comparison-model complexity: O(n^2 log n).
    No exact fallback is used.
    """
    best=omega_q2_strict(D)
    best_cost=tour_length(D,best)

    # Fixed number of bridge attempts.
    for _ in range(4):
        candidate,_=_q3_plateau_bridge(
            D,best,
            depth=2,
            beam_width=64,
            neutral_cap=96
        )
        if candidate is None:
            break

        c=tour_length(D,candidate)
        if c<best_cost-1e-12:
            best,best_cost=candidate,c
        else:
            break

    return best


# ============================================================
# OMEGA-Q5
#
# Q5 attacks two failure modes:
#   1) neutral plateaus (fixed constant-depth bridge)
#   2) positive barriers / coordinated multi-edge exchanges
#      via a sparse negative-cycle cycle-cover jump, followed by
#      a bounded number of subtour-merging successor swaps.
#
# Generic comparison-model complexity:
#   O(n^2 log n)
#
# Integer/fixed-width fast path:
#   O(n^2) word-RAM target using stable radix ordering of the
#   O(n^2) regret scores.
#
# No exact fallback is used by omega_q5 / omega_q5_int.
# ============================================================

import heapq

def _q5_successor_maps(tour):
    succ={}
    pred={}
    pos={}
    n=len(tour)
    for i,u in enumerate(tour):
        v=tour[(i+1)%n]
        succ[u]=v
        pred[v]=u
        pos[u]=i
    return succ,pred,pos

def _q5_succ_from_tour(tour):
    return {tour[i]:tour[(i+1)%len(tour)] for i in range(len(tour))}

def _q5_succ_cost(D,succ):
    return float(sum(D[u,v] for u,v in succ.items()))

def _q5_permutation_cycles(succ):
    seen=set()
    cycles=[]
    for s in succ:
        if s in seen:
            continue
        cyc=[]
        u=s
        while u not in seen:
            seen.add(u)
            cyc.append(u)
            u=succ[u]
        cycles.append(cyc)
    return cycles

def _q5_succ_to_tour(succ,start=0):
    n=len(succ)
    out=[start]
    seen={start}
    u=start
    for _ in range(n-1):
        u=succ[u]
        if u in seen:
            return None
        seen.add(u)
        out.append(u)
    return out if succ[u]==start else None

def _q5_build_exchange_digraph(D,tour,K=6):
    """
    Sparse assignment-exchange digraph.
    Arc i->j means tail tour[i] takes the current successor of tour[j].
    K is a fixed constant, so total arcs = O(n).
    """
    n=len(tour)
    succ,_,_=_q5_successor_maps(tour)
    arcs=[[] for _ in range(n)]

    for i,u in enumerate(tour):
        old=succ[u]
        cand=[]
        for j,w in enumerate(tour):
            if i==j:
                continue
            head=succ[w]
            if head==u:
                continue
            delta=float(D[u,head]-D[u,old])
            cand.append((delta,j))
        best=heapq.nsmallest(K,cand,key=lambda x:x[0])
        arcs[i]=[(j,delta) for delta,j in best]
    return arcs

def _q5_negative_cycle(arcs):
    """
    Bellman-Ford from an implicit zero-cost super-source.
    With E=O(n) (constant K), this is O(n^2).
    """
    n=len(arcs)
    dist=[0.0]*n
    parent=[-1]*n
    x=-1

    for _ in range(n):
        x=-1
        for u in range(n):
            du=dist[u]
            for v,w in arcs[u]:
                nd=du+w
                if nd<dist[v]-1e-12:
                    dist[v]=nd
                    parent[v]=u
                    x=v
        if x==-1:
            return None

    y=x
    for _ in range(n):
        y=parent[y]
        if y<0:
            return None

    cyc=[y]
    cur=parent[y]
    while cur!=y and cur>=0 and len(cyc)<=n+1:
        cyc.append(cur)
        cur=parent[cur]
    if cur!=y:
        return None
    cyc.reverse()
    return cyc

def _q5_apply_cycle_cover(tour,cyc):
    succ=_q5_succ_from_tour(tour)
    new=dict(succ)
    k=len(cyc)

    for r,i in enumerate(cyc):
        j=cyc[(r+1)%k]
        u=tour[i]
        w=tour[j]
        new[u]=succ[w]
    return new

def _q5_patch_cover(D,succ,patch_cap=12):
    """
    Merge a bounded number of subtours.
    Each merge scans O(n^2) pairs; patch_cap is a fixed constant.
    """
    s=dict(succ)
    moves=[]

    for _ in range(patch_cap):
        cycles=_q5_permutation_cycles(s)
        if len(cycles)==1:
            return _q5_succ_to_tour(s,0),moves

        comp={}
        for ci,C in enumerate(cycles):
            for u in C:
                comp[u]=ci

        nodes=list(s)
        best=None

        for ai,u in enumerate(nodes):
            a=s[u]
            for v in nodes[ai+1:]:
                if comp[u]==comp[v]:
                    continue
                b=s[v]
                delta=float(D[u,b]+D[v,a]-D[u,a]-D[v,b])
                if best is None or delta<best[0]:
                    best=(delta,u,v,a,b)

        if best is None:
            break

        delta,u,v,a,b=best
        s[u],s[v]=b,a
        moves.append((u,v,delta))

    return None,moves

def _q5_jump(D,tour,K_values=(4,6,8,12),patch_cap=12):
    base=tour_length(D,tour)
    best=tour
    best_cost=base
    best_meta=None

    for K in K_values:
        arcs=_q5_build_exchange_digraph(D,tour,K=K)
        cyc=_q5_negative_cycle(arcs)

        if cyc is None:
            continue

        cover=_q5_apply_cycle_cover(tour,cyc)
        cover_cost=_q5_succ_cost(D,cover)
        patched,moves=_q5_patch_cover(D,cover,patch_cap=patch_cap)

        if patched is None:
            continue

        c=tour_length(D,patched)
        if c<best_cost-1e-12:
            best=patched
            best_cost=c
            best_meta={
                "K":K,
                "negative_cycle_length":len(cyc),
                "cover_cost":cover_cost,
                "patch_moves":moves,
                "gain":base-c,
            }

    return best,best_meta

def _q5_plateau_stage(D,tour):
    """
    Constant-depth neutral bridge.
    Depth/beam/cap are fixed and do not grow with n.
    """
    best=tour
    best_cost=tour_length(D,best)

    for _ in range(4):
        cand,_=_q3_plateau_bridge(
            D,best,
            depth=8,
            beam_width=96,
            neutral_cap=128
        )
        if cand is None:
            break

        c=tour_length(D,cand)
        if c<best_cost-1e-12:
            best,best_cost=cand,c
        else:
            break

    return best

def omega_q5(D):
    """
    Generic Q5.

    Starts from OMEGA-Q2 STRICT, not Q4.
    Therefore it does not inherit Q4's O(n^3) plateau closure.

    Comparison-model complexity: O(n^2 log n).
    """
    D=np.asarray(D,dtype=np.float64)

    best=omega_q2_strict(D)
    best_cost=tour_length(D,best)

    for _ in range(3):
        cand=_q5_plateau_stage(D,best)
        c=tour_length(D,cand)
        if c<best_cost-1e-12:
            best,best_cost=cand,c

        cand,_=_q5_jump(D,best,K_values=(4,6,8,12),patch_cap=12)
        c=tour_length(D,cand)
        if c<best_cost-1e-12:
            best,best_cost=cand,c
            continue

        cand=_q5_plateau_stage(D,best)
        c=tour_length(D,cand)
        if c<best_cost-1e-12:
            best,best_cost=cand,c
        else:
            break

    return best

# ------------------------------------------------------------
# Integer O(n^2) fast path
# ------------------------------------------------------------

def _q5_radix_argsort_int64(keys):
    """
    Stable LSD radix sort on signed 64-bit keys.
    Eight base-256 passes => O(M) for fixed-width words.
    """
    keys=np.asarray(keys,dtype=np.int64)
    m=len(keys)
    u=keys.view(np.uint64) ^ np.uint64(1<<63)
    idx=np.arange(m,dtype=np.int64)
    tmp=np.empty_like(idx)

    for shift in range(0,64,8):
        byte=((u[idx]>>np.uint64(shift))&np.uint64(0xFF)).astype(np.int16)
        counts=np.bincount(byte,minlength=256)

        pos=np.empty(256,dtype=np.int64)
        total=0
        for b in range(256):
            pos[b]=total
            total+=int(counts[b])

        for q in range(m):
            b=int(byte[q])
            tmp[pos[b]]=idx[q]
            pos[b]+=1

        idx,tmp=tmp,idx

    return idx

def _q5_omega_q2_int(D,beta=0.5,two_opt_pass=False):
    D=np.asarray(D,dtype=np.int64)
    n=D.shape[0]

    W=D.copy()
    INF=np.iinfo(np.int64).max//8
    np.fill_diagonal(W,INF)

    two=np.argpartition(W,kth=1,axis=1)[:,:2]
    vals=np.take_along_axis(W,two,axis=1)
    ord2=np.argsort(vals,axis=1)

    first_idx=two[np.arange(n),ord2[:,0]]
    first_val=vals[np.arange(n),ord2[:,0]]
    second_val=vals[np.arange(n),ord2[:,1]]

    iu,iv=np.triu_indices(n,1)
    w=D[iu,iv]

    alt_u=np.where(first_idx[iu]==iv,second_val[iu],first_val[iu])
    alt_v=np.where(first_idx[iv]==iu,second_val[iv],first_val[iv])

    beta8=int(round(beta*8))
    score8=8*w-beta8*((alt_u-w)+(alt_v-w))
    order=_q5_radix_argsort_int64(score8)

    deg=np.zeros(n,dtype=np.int8)
    dsu=DSU(n)
    edges=[]

    for idx in order:
        u=int(iu[idx])
        v=int(iv[idx])

        if deg[u]>=2 or deg[v]>=2:
            continue
        if dsu.find(u)==dsu.find(v):
            continue

        edges.append((u,v))
        deg[u]+=1
        deg[v]+=1
        dsu.union(u,v)

        if len(edges)==n-1:
            break

    ends=np.flatnonzero(deg==1)
    edges.append((int(ends[0]),int(ends[1])))
    tour=_cycle_from_edges(n,edges)

    if two_opt_pass:
        tour=_one_full_two_opt_pass(D.astype(np.float64),tour)

    return tour

def _q5_q2_strict_int(D):
    Dint=np.asarray(D,dtype=np.int64)
    Df=Dint.astype(np.float64)

    candidates=[]

    for beta in (-1.0,-0.5,0.0,0.125,0.25,0.5,0.75,1.0,1.5,2.0,3.0,4.0):
        candidates.append(_q5_omega_q2_int(Dint,beta=beta,two_opt_pass=False))

    n=len(Dint)
    starts=list(range(min(n,16)))

    for s in starts:
        candidates.append(_nearest_neighbor_start_q2(Df,s))

    for s in starts[:8]:
        candidates.append(_insertion_q2(Df,s,"farthest"))
        candidates.append(_insertion_q2(Df,s,"nearest"))

    best=None
    best_cost=float("inf")

    for t in candidates:
        t=_polish_q2(Df,t,rounds=8)
        c=tour_length(Df,t)

        if c<best_cost:
            best,best_cost=t,c

    for kt in _fixed_kicks_q2(best)[:48]:
        t=_polish_q2(Df,kt,rounds=10)
        c=tour_length(Df,t)

        if c<best_cost:
            best,best_cost=t,c

    snap=best

    for kt in _fixed_kicks_q2(snap)[:24]:
        t=_polish_q2(Df,kt,rounds=10)
        c=tour_length(Df,t)

        if c<best_cost:
            best,best_cost=t,c

    return best

def omega_q5_int(D):
    """
    Integer/fixed-width Q5.

    Word-RAM target complexity: O(n^2).

    Assumptions:
      - integer edge weights fit fixed-width int64
      - radix sort is treated as linear in the number of fixed-width keys
      - all Q5 caps/depths are fixed constants
    """
    Dint=np.asarray(D,dtype=np.int64)
    Df=Dint.astype(np.float64)

    best=_q5_q2_strict_int(Dint)
    best_cost=tour_length(Df,best)

    for _ in range(3):
        cand=_q5_plateau_stage(Df,best)
        c=tour_length(Df,cand)

        if c<best_cost-1e-12:
            best,best_cost=cand,c

        cand,_=_q5_jump(Df,best,K_values=(4,6,8,12),patch_cap=12)
        c=tour_length(Df,cand)

        if c<best_cost-1e-12:
            best,best_cost=cand,c
            continue

        break

    return best


# ============================================================
# OMEGA-Q6
#
# Q5 failure mode:
#   _q5_negative_cycle() returned one arbitrary Bellman-Ford
#   negative cycle. Another negative cycle in the same exchange
#   graph could produce a better valid Hamiltonian tour.
#
# Q6:
#   Enumerate a bounded family of negative cycles via dynamic
#   programming on the sparse exchange graph, then evaluate them
#   after cycle-cover application + bounded patching.
#
# With K, max_cycle_len, top_m and outer rounds fixed:
#
#   generic comparison model: O(n^2 log n)
#   fixed-width integer word-RAM path: O(n^2)
#
# No Held-Karp / exact fallback is used by omega_q6 or omega_q6_int.
# ============================================================

def _q6_canonical_cycle(cyc):
    cyc=list(cyc)
    m=len(cyc)
    rots=[tuple(cyc[i:]+cyc[:i]) for i in range(m)]
    rev=list(reversed(cyc))
    rots += [tuple(rev[i:]+rev[:i]) for i in range(m)]
    return min(rots)

def _q6_negative_cycles_dp(arcs,max_len=12,top_m=128):
    """
    Return many negative simple cycles from the sparse exchange graph.

    For each start s and each exact path length ell<=max_len,
    keep the cheapest simple path found to each endpoint.

    max_len is a fixed constant and the exchange graph has O(n)
    arcs for fixed K, so the total work over all starts is O(n^2).
    """
    n=len(arcs)
    candidates={}

    for s in range(n):
        dp={s:(0.0,[s])}

        for ell in range(1,max_len+1):
            ndp={}

            for u,(cost,path) in dp.items():
                for v,w in arcs[u]:
                    if v in path and v!=s:
                        continue

                    nc=cost+w
                    npath=path+[v]

                    old=ndp.get(v)
                    if old is None or nc<old[0]:
                        ndp[v]=(nc,npath)

                    if v==s and ell>=2 and nc< -1e-12:
                        cyc=path[:]

                        if len(set(cyc))==len(cyc):
                            key=_q6_canonical_cycle(cyc)

                            if key not in candidates or nc<candidates[key][0]:
                                candidates[key]=(nc,list(key))

            dp=ndp

            if not dp:
                break

    return sorted(candidates.values(),key=lambda x:x[0])[:top_m]

def _q6_evaluate_cycles(D,tour,cycles):
    base=tour_length(D,tour)
    best=tour
    best_cost=base
    best_meta=None

    for cyc_cost,cyc in cycles:
        cover=_q5_apply_cycle_cover(tour,cyc)
        cover_cost=_q5_succ_cost(D,cover)

        patched,moves=_q5_patch_cover(D,cover,patch_cap=12)

        if patched is None:
            continue

        c=tour_length(D,patched)

        if c<best_cost-1e-12:
            best=patched
            best_cost=c
            best_meta={
                "cycle":cyc,
                "exchange_delta":cyc_cost,
                "cover_cost":cover_cost,
                "patched_cost":c,
                "patch_moves":moves,
                "gain":base-c,
            }

    return best,best_meta

def _q6_multi_cycle_stage(D,tour,max_cycle_len=12,top_m=128):
    best=tour
    best_cost=tour_length(D,best)

    for K in (4,6,8,12):
        arcs=_q5_build_exchange_digraph(D,best,K=K)
        cycles=_q6_negative_cycles_dp(
            arcs,
            max_len=max_cycle_len,
            top_m=top_m
        )

        cand,_=_q6_evaluate_cycles(D,best,cycles)
        c=tour_length(D,cand)

        if c<best_cost-1e-12:
            return cand

    return best

def omega_q6(D,max_cycle_len=12,top_m=128):
    """
    Generic Q6.

    Q5 base + bounded multi-negative-cycle selector.

    Comparison-model complexity:
        O(n^2 log n)
    """
    D=np.asarray(D,dtype=np.float64)

    best=omega_q5(D)
    best_cost=tour_length(D,best)

    for _ in range(3):
        cand=_q6_multi_cycle_stage(
            D,best,
            max_cycle_len=max_cycle_len,
            top_m=top_m
        )

        c=tour_length(D,cand)

        if c<best_cost-1e-12:
            best,best_cost=cand,c
        else:
            break

    return best

def omega_q6_int(D,max_cycle_len=12,top_m=128):
    """
    Fixed-width integer Q6.

    omega_q5_int supplies the O(n^2) integer base.
    The Q6 multi-cycle stage is O(n^2) for fixed K/L/top_m.

    Word-RAM target complexity:
        O(n^2)
    """
    Dint=np.asarray(D,dtype=np.int64)
    Df=Dint.astype(np.float64)

    best=omega_q5_int(Dint)
    best_cost=tour_length(Df,best)

    for _ in range(3):
        cand=_q6_multi_cycle_stage(
            Df,best,
            max_cycle_len=max_cycle_len,
            top_m=top_m
        )

        c=tour_length(Df,cand)

        if c<best_cost-1e-12:
            best,best_cost=cand,c
        else:
            break

    return best
