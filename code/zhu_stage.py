#!/usr/bin/env python3
# zhu_stage.py — Etappenweise Berechnung des Zertifikats (gleiche Mathematik wie zhu_cert.py)
#   part:     python3 zhu_stage.py part  L T N panels prec q0 q1 out.txt   (Teilsumme K über Knoten q0..q1-1)
#   finalize: python3 zhu_stage.py final L T N panels prec part1.txt part2.txt ... > cert.json
# Teilsummen werden als Ball (Mittelpunkt, Radius) gespeichert; Summe in Arb ist weiterhin rigoros.
#   Parität: Umgebungsvariable PARITY=even (Standard) oder PARITY=odd. Gerade/ungerade Sektoren sind bezüglich Q
#   orthogonal (translationsinvarianter, gerader Kern); volle Positivität auf [−L, L] = beide Sektoren.
#   Moden m = 2k + σ (σ = 0 gerade, 1 ungerade): ê_k(t) = c_k·i^m·j_m(Lt), c_k = sqrt((2m+1)/(2L))·2L,
#   Pol-Term P(f) = 2 f̂(i/2) f̂(−i/2) = 2(−1)^σ (Σ a_k c_k i_m(L/2))²  (im ungeraden Sektor NEGATIV).
import sys, os, time, json
from flint import arb, acb, arb_mat, ctx

ORDER = 32
SIGMA = 1 if os.environ.get("PARITY", "even") == "odd" else 0
def mo(k): return 2*k + SIGMA

def setup(L, T, N, panels, prec):
    ctx.prec = prec
    L = arb(L); T = arb(T)
    def mangoldt(n):
        for p in range(2, n+1):
            if n % p == 0:
                m = n
                while m % p == 0: m //= p
                return arb(p).log() if m == 1 else None
        return None
    comb = []; n = 2
    while arb(n).log() < 2*L:
        lam = mangoldt(n)
        if lam is not None: comb.append((arb(n).log(), 2*lam/arb(n).sqrt()))
        n += 1
    A_L = sum((a for _, a in comb), arb(0))
    beta = (T/(2*arb.pi())).log() - 1/T - A_L
    assert beta > 0
    gl = [arb.legendre_p_root(ORDER, k, weight=True) for k in range(ORDER)]
    h = T/panels
    nodes, weights = [], []
    for p in range(panels):
        for x, w in gl: nodes.append(h*p + h*(x+1)/2); weights.append(w*h/2)
    c = [((2*mo(k)+1)/(2*L)).sqrt()*2*L for k in range(N)]
    return dict(L=L, T=T, N=N, panels=panels, prec=prec, comb=comb, A_L=A_L, beta=beta,
                nodes=nodes, weights=weights, c=c)

def sj(nn, x): return (arb.pi()/(2*x)).sqrt()*x.bessel_j(arb(nn)+arb(1)/2)
def sj_all(x, nmax, prec):
    # j_0..j_nmax am exakten Mittelpunkt per Aufwärtsrekurrenz (2048 Bit), Unschärfe von x über |j_n'| ≤ ½
    # (aus j_n(x) = ½(−i)^n ∫_{−1}^{1} P_n(s) e^{ixs} ds) wieder aufgeschlagen; Fallback auf direkte Auswertung,
    # sobald der Rekurrenz-Ball zu breit wird. Rigoros.
    xm = x.mid(); dx = x.rad()
    old = ctx.prec; ctx.prec = 2048
    js = [xm.sin()/xm, xm.sin()/(xm*xm) - xm.cos()/xm]
    for n in range(1, nmax): js.append((2*n+1)/xm*js[-1] - js[-2])
    ctx.prec = old
    tol = arb(2)**(-prec+10); out = []
    for n in range(nmax+1):
        v = js[n]
        out.append(+v + arb(0, dx) if v.rad() < tol else sj(n, x))
    return out

def si(nn, x): return (arb.pi()/(2*x)).sqrt()*x.bessel_i(arb(nn)+arb(1)/2)

def part(S, q0, q1, out):
    L, N, c, beta = S["L"], S["N"], S["c"], S["beta"]
    K = arb_mat(N, N); CH = 512; t0 = time.time()
    for a in range(q0, q1, CH):
        qs = range(a, min(a+CH, q1))
        E = arb_mat(len(qs), N); EW = arb_mat(len(qs), N)
        for r, q in enumerate(qs):
            t = S["nodes"][q]
            psi = acb(arb(1)/4, t/2).digamma().real - arb.pi().log()
            for fr, am in S["comb"]: psi -= am*(t*fr).cos()
            wt = S["weights"][q]*(psi - beta)/arb.pi()
            jl = sj_all(L*t, 2*N + 1, S["prec"])
            for k in range(N):
                v = c[k]*jl[mo(k)]*(-1 if k % 2 else 1)
                E[r, k] = v; EW[r, k] = v*wt
        K = K + E.transpose()*EW
        print(f"  [{q0}-{q1}] {a+len(qs)-q0}/{q1-q0}  {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
    dig = int(S["prec"]*0.31)
    with open(out, "w") as f:
        for i in range(N):
            for j in range(i, N):
                e = K[i, j]
                f.write(f"{i} {j} {e.mid().str(dig, radius=False)} {e.rad().str(5, radius=False)}\n")

def final(S, parts):
    import mpmath as mp
    from flint import acb_mat
    L, T, N, beta, c, prec = S["L"], S["T"], S["N"], S["beta"], S["c"], S["prec"]
    t0 = time.time()
    K = [[arb(0)]*N for _ in range(N)]
    for pth in parts:
        for line in open(pth):
            i, j, m, r = line.split(); i = int(i); j = int(j)
            # Rundung beim Speichern rigoros aufschlagen: Radius (5 Stellen) ×1,001 und Mittelpunkt-Rundung
            # (≥ 119 Stellen) mit relativem Zuschlag 1e−110.
            mv = arb(m)
            K[i][j] = K[i][j] + mv + arb(0, arb(r).upper()*arb('1.001') + abs(mv).upper()*arb('1e-110') + arb('1e-300'))
    for i in range(N):
        for j in range(i): K[i][j] = K[j][i]
    pv = [c[k]*si(mo(k), L/2) for k in range(N)]
    sgn = -1 if SIGMA else 1
    M = arb_mat(N, N)
    for i in range(N):
        for j in range(N):
            M[i, j] = K[i][j] + sgn*2*pv[i]*pv[j] + (beta if i == j else 0)
    # Tail-Schranken
    A_L = S["A_L"]
    psi_bound = abs(acb(arb(1)/4, 0).digamma().real) + abs(acb(arb(1)/4, T/2).digamma().real) + arb.pi().log() + A_L + beta
    Sb = T/arb.pi()*psi_bound
    def dfact(m):
        r = arb(1)
        for i in range(1, 2*m+2, 2): r *= i
        return r
    tail = [((2*mo(k)+1)/(2*L)).sqrt()*2*L*(L*T)**mo(k)/dfact(mo(k)) for k in range(N, N+400)]
    assert tail[-1]/tail[-2] < 0.5
    tail_norm2 = sum((b*b for b in tail), arb(0))*2
    head_sup = sum((ck*ck for ck in c), arb(0)).sqrt()
    eps_B = Sb*head_sup*tail_norm2.sqrt()
    eps_D = Sb*tail_norm2
    pole_tail = sum(((((2*mo(k)+1)/(2*L)).sqrt()*2*L*(L/2)**mo(k)*(L/2).exp()/dfact(mo(k)))**2 for k in range(N, N+200)), arb(0)).sqrt()*2
    eps_B += 2*sum((p*p for p in pv), arb(0)).sqrt()*pole_tail
    # Tail-Diagonalblock: R(t) ≥ (β* − ε_D − 2‖pv_tail‖²)‖t‖² (Pol-Term im ungeraden Sektor negativ, im geraden ≥ 0).
    # Mit R(h+t) ≥ λ‖h‖² − 2ε_B‖h‖‖t‖ + μ‖t‖² folgt R ≥ min(λ, μ) − ε_B; geprüft wird μ > λ.
    mu_tail = beta - eps_D - pole_tail*pole_tail
    # Rigorose Quadraturschranke (Bernstein-Ellipse, Binet-II-Digamma-Schranke)
    bt = arb('0.4'); hh = T/S["panels"]; r = (bt*bt + (hh/2)**2).sqrt()
    quad_ok = bool(r < arb('0.5')); eps_quad = arb('inf')
    if quad_ok:
        rho = (bt + r)/(hh/2)
        geo = arb(64)/15*rho**(-2*ORDER)/(rho*rho - 1)*(hh/2)
        csum = sum((ck*ck for ck in c), arb(0))*(2*bt*L).exp()
        tot = arb(0)
        for pnl in range(S["panels"]):
            tc = hh*pnl + hh/2
            re_lo = arb(1)/4 - r/2
            im_lo = (arb(tc) - r)/2 if arb(tc) > r else arb(0)
            im_hi = (arb(tc) + r)/2
            w_hi = ((1 + arb(1)/4 + r/2)**2 + im_hi**2).sqrt(); w_lo = arb(1) + re_lo
            z_lo = (re_lo**2 + im_lo**2).sqrt()
            pb = (w_hi.log() + arb.pi()/2 + 1/(2*w_lo) + arb(1)/9 + 1/z_lo) + arb.pi().log() + abs(beta)
            for fr, am in S["comb"]: pb += am*(bt*fr).cosh()
            tot += pb.upper()
        eps_quad = geo*tot*csum/arb.pi()
    # Kopfblock: Kandidat aus Flint-Eigenlöser, Rump-Verifikation
    mp.mp.dps = int(prec*0.29)
    ev = acb_mat(arb_mat(N, N, [M[i, j].mid() for i in range(N) for j in range(N)])).eig(algorithm="approx")
    lam_mid = mp.mpf(min(e.real for e in ev).mid().str(int(prec*0.3), radius=False))
    res = dict(parity="odd" if SIGMA else "even", L=float(L.mid()), T_sharp=float(T.mid()), N=N, panels=S["panels"], prec_bits=prec,
               beta=float(beta.mid()), lambda_mid=mp.nstr(lam_mid, 8),
               max_radius_M=max(float(M[i, j].rad()) for i in range(N) for j in range(N)),
               eps_B=float(eps_B.upper()), eps_D=float(eps_D.upper()),
               eps_quad=float(eps_quad.upper()) if quad_ok else None, quad_rigorous=quad_ok)
    if lam_mid <= 0:
        res["verified"] = False; res["note"] = "Kandidat-Eigenwert nicht positiv"
        print(json.dumps(res, indent=1)); return
    Mm = mp.matrix(N, N)
    for i in range(N):
        for j in range(N): Mm[i, j] = mp.mpf(M[i, j].mid().str(int(prec*0.3), radius=False))
    c_mp = lam_mid*mp.mpf("0.99")
    Lc = mp.cholesky(Mm - c_mp*mp.eye(N))
    dig = int(prec*0.3)
    Lm = arb_mat(N, N, [arb(mp.nstr(Lc[i, j], dig, strip_zeros=False)).mid() if j <= i else arb(0) for i in range(N) for j in range(N)])
    cand = arb(mp.nstr(c_mp, dig)).mid()
    X = Lm*Lm.transpose(); fro2 = arb(0)
    for i in range(N):
        for j in range(N):
            u = abs(M[i, j] - X[i, j] - (cand if i == j else 0)).upper(); fro2 += u*u
    resid = fro2.sqrt(); head_lower = cand - resid
    res["residual"] = float(resid.upper()); res["head_lower"] = float(head_lower.lower())
    res["verified"] = bool(head_lower > 0)
    res["tail_block_lower"] = float(mu_tail.lower())
    res["tail_block_ok"] = bool(mu_tail > head_lower)
    if res["verified"] and quad_ok and res["tail_block_ok"]:
        res["certified_lower_bound"] = float((head_lower - eps_B - eps_quad).lower())
    res["final_s"] = round(time.time()-t0, 1)
    print(json.dumps(res, indent=1))

if __name__ == "__main__":
    mode = sys.argv[1]
    L, T, N, P, prec = float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
    S = setup(L, T, N, P, prec)
    if mode == "part":
        part(S, int(sys.argv[7]), int(sys.argv[8]), sys.argv[9])
    else:
        final(S, sys.argv[7:])
