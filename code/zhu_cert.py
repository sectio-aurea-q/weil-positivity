#!/usr/bin/env python3
# zhu_cert.py — Reproduktion/Erweiterung des Weil-Positivitäts-Zertifikats nach Zhu (arXiv:2608.24827)
# LINDBLAD-RIEMANN A.6 / P.6 · sectio-aurea-q · MEGALODON Research · 2026
#
# Weil-Form für reelles gerades f, supp f ⊂ [−L, L]  (f̂(r) = ∫ f e^{iru} du):
#   Q(f) = 2 f̂(i/2)² + (1/π) ∫_0^∞ |f̂(t)|² Ψ_L(t) dt,
#   Ψ_L(t) = Re ψ(¼+it/2) − log π − Σ_{log n<2L} 2Λ(n)/√n cos(t log n).
# Reduktion (Zhu): für T♯ > T₁ = 2π e^{A_L} gilt Ψ_L(t) ≥ β* := log(T♯/2π) − 1/T♯ − A_L > 0 für t ≥ T♯, also
#   Q(f) ≥ R(f) = 2 f̂(i/2)² + (1/π)∫_0^{T♯} (Ψ_L − β*)|f̂|² dt + β*‖f‖².
# Basis: gerade normierte Legendre-Polynome e_k(u) = sqrt((4k+1)/(2L)) P_{2k}(u/L), k = 0..∞,
#   ê_k(t) = sqrt((4k+1)/(2L)) · 2L · (−1)^k j_{2k}(L t),   ê_k(i/2) = sqrt((4k+1)/(2L)) · 2L · i_{2k}(L/2).
# Zertifikat: M_N (Kopfblock) in Arb-Ball-Arithmetik; LDLᵀ von M_N − c·I mit garantiert positiven Pivots
#   ⇒ λ_min(M_N) > c. Tail-Kopplung über |j_n(x)| ≤ x^n/(2n+1)!!.
#
# STATUS: Der Quadraturfehler (Gauß-Legendre über [0,T♯]) ist hier NUMERISCH geschätzt (Panelverdopplung),
# noch nicht rigoros (Bernstein-Ellipse wie Zhu Lemma 5.1). Bis dahin ist das Ergebnis eine
# hochpräzise Reproduktion, KEIN vollständiges Zertifikat.
import sys, time, json
from flint import arb, acb, arb_mat, ctx

def run(L, T, N, panels, order=32, prec=320, verbose=True):
    ctx.prec = prec
    L = arb(L); T = arb(T); t0 = time.time()
    # --- Primzahl-Kamm ---
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
    T1 = 2*arb.pi()*A_L.exp()
    beta = (T/(2*arb.pi())).log() - 1/T - A_L
    assert beta > 0, "T♯ muss > T₁ sein"
    # --- Gauß-Legendre-Knoten (rigoros eingeschlossen) ---
    gl = [arb.legendre_p_root(order, k, weight=True) for k in range(order)]
    nodes, weights = [], []
    h = T/panels
    for p in range(panels):
        a = h*p
        for x, w in gl:
            nodes.append(a + h*(x+1)/2); weights.append(w*h/2)
    Q = len(nodes)
    # --- Basis im Frequenzraum ---
    c = [((4*k+1)/(2*L)).sqrt()*2*L for k in range(N)]
    def sj(nn, x):  # sphärische Bessel j_n
        return (arb.pi()/(2*x)).sqrt()*x.bessel_j(arb(nn)+arb(1)/2)
    K = arb_mat(N, N)
    CH = 1024
    for q0 in range(0, Q, CH):
        qs = list(range(q0, min(q0+CH, Q)))
        E = arb_mat(len(qs), N); EW = arb_mat(len(qs), N)
        for r, q in enumerate(qs):
            t = nodes[q]
            psi = acb(arb(1)/4, t/2).digamma().real - arb.pi().log()
            for fr, am in comb: psi -= am*(t*fr).cos()
            wt = weights[q]*(psi - beta)/arb.pi()
            for k in range(N):
                v = c[k]*sj(2*k, L*t)*(-1 if k % 2 else 1)
                E[r, k] = v; EW[r, k] = v*wt
        K = K + E.transpose()*EW
        if verbose and (q0//CH) % 5 == 0: print(f"  Knoten {q0+len(qs)}/{Q}  ({time.time()-t0:.0f}s)", flush=True)
    # Pol-Vektor
    def si(nn, x): return (arb.pi()/(2*x)).sqrt()*x.bessel_i(arb(nn)+arb(1)/2)
    pv = [c[k]*si(2*k, L/2) for k in range(N)]
    M = arb_mat(N, N)
    for i in range(N):
        for j in range(N):
            M[i, j] = K[i, j] + 2*pv[i]*pv[j] + (beta if i == j else 0)
    t_build = time.time()-t0
    if verbose:
        rmax = max(float(M[i, j].rad()) for i in range(N) for j in range(N))
        print(f"max. Radius in M: {rmax:.2e}", flush=True)
    # --- Tail-Schranken (Moden k ≥ N) ---
    # |ê_k(t)| ≤ c_k (L t)^{2k}/(4k+1)!! ≤ c_k (L T)^{2k}/(4k+1)!!  für t ≤ T♯
    # Σ_q |w_q (Ψ−β)/π| ≤ (T/π)·sup|Ψ−β|; sup|Ψ−β| ≤ |Re ψ| + log π + A_L + β, Re ψ auf [0,T] durch Randwerte
    psi_bound = abs(acb(arb(1)/4, 0).digamma().real) + abs(acb(arb(1)/4, T/2).digamma().real) + arb.pi().log() + A_L + beta
    S = T/arb.pi()*psi_bound
    def dfact(m):  # (2m+1)!!
        r = arb(1)
        for i in range(1, 2*m+2, 2): r *= i
        return r
    tail = []
    for k in range(N, N+400):
        b = ((4*k+1)/(2*L)).sqrt()*2*L*(L*T)**(2*k)/dfact(2*k)
        tail.append(b)
    # geometrischer Rest ab k=N+400: Quotient < 1/2 vorausgesetzt (geprüft)
    ratio = tail[-1]/tail[-2]
    assert ratio < 0.5
    tail_norm2 = sum((b*b for b in tail), arb(0))*2
    tail_norm = tail_norm2.sqrt()
    head_sup = sum((ck*ck for ck in c), arb(0)).sqrt()          # |j_n| ≤ 1
    eps_B = S*head_sup*tail_norm
    eps_D = S*tail_norm2
    pole_tail = sum(((((4*k+1)/(2*L)).sqrt()*2*L*(L/2)**(2*k)*(L/2).exp()/dfact(2*k))**2 for k in range(N, N+200)), arb(0)).sqrt()*2
    pole_head = sum((p*p for p in pv), arb(0)).sqrt()
    eps_B += 2*pole_head*pole_tail
    # --- Rigorose Quadraturschranke (Gauß–Legendre, Bernstein-Ellipse; vgl. Zhu Lemma 5.1, Trefethen) ---
    # Pro Panel [a, a+h] mit n Knoten: |Fehler| ≤ (h/2)·(64/15)·Mf·ρ^{−2n}/(ρ²−1), falls der Integrand in der
    # Ellipse E_ρ (Brennpunkte a, a+h) analytisch und durch Mf beschränkt ist.
    # Wahl: Halbachse b_t = 0.4 in t-Einheiten (Digamma-Pole bei t = 2i(k+¼), nächster bei i/2);
    # die Ellipse liegt in der Kreisscheibe Radius r = sqrt(b_t² + (h/2)²) um die Panelmitte; verlangt r < 0.5.
    # |ê_k(z)| ≤ c_k·e^{L·|Im z|} ≤ c_k·e^{0.4 L}  (aus j_n(z) = (−i)^n/2 ∫P_n e^{izx}, |P_n| ≤ 1).
    bt = arb('0.4'); hh = T/panels
    r = (bt*bt + (hh/2)**2).sqrt()
    eps_quad = arb(0); quad_ok = bool(r < arb('0.5'))
    if quad_ok:
        rho = (bt + r)/(hh/2)
        geo = arb(64)/15 * rho**(-2*order)/(rho*rho - 1) * (hh/2)
        csum = sum((ck*ck for ck in c), arb(0)) * (2*bt*L).exp()   # Σ c_k² · e^{0.8 L}
        tot = arb(0)
        def psi_sup(tc):
            # Rigorose Schranke für |ψ(z)|, z = ¼ + iζ/2, |ζ − tc| ≤ r:
            # ψ(z) = ψ(z+1) − 1/z;  für Re w ≥ 1 (Binet II): |ψ(w)| ≤ log|w| + π/2 + 1/(2|w|) + 1/9,
            # denn |t² + w²| ≥ 3/4 für Re w ≥ 1 und ∫_0^∞ 2t/(e^{2πt} − 1) dt = 1/12.
            re_lo = arb(1)/4 - r/2
            im_abs_lo = max(arb(0), (arb(tc) - r)/2) if arb(tc) > r else arb(0)
            im_abs_hi = (arb(tc) + r)/2
            w_abs_hi = ((1 + arb(1)/4 + r/2)**2 + im_abs_hi**2).sqrt()
            w_abs_lo = arb(1) + re_lo
            z_abs_lo = (re_lo**2 + im_abs_lo**2).sqrt()
            return (w_abs_hi.log() + arb.pi()/2 + 1/(2*w_abs_lo) + arb(1)/9 + 1/z_abs_lo).upper()
        for pnl in range(panels):
            tc = hh*pnl + hh/2
            psi_b = psi_sup(tc) + arb.pi().log() + abs(beta)
            for fr, am in comb: psi_b += am*(bt*fr).cosh()
            tot += psi_b.upper()
        eps_quad = geo*tot*csum/arb.pi()          # Frobenius-Schranke: |E_mn| ≤ K c_m c_n ⇒ ‖E‖_F ≤ K Σ c_k²
    # --- Kopfblock: Mittelpunkt-Eigenwert als Kandidat, dann rigorose LDLᵀ ---
    import mpmath as mp
    mp.mp.dps = int(prec*0.29)
    Mm = mp.matrix(N, N)
    for i in range(N):
        for j in range(N): Mm[i, j] = mp.mpf(M[i, j].mid().str(int(prec*0.3), radius=False))
    from flint import acb_mat
    ev = acb_mat(arb_mat(N, N, [M[i, j].mid() for i in range(N) for j in range(N)])).eig(algorithm="approx")
    lam_mid = mp.mpf(min(e.real for e in ev).mid().str(int(prec*0.3), radius=False))
    cand = arb(mp.nstr(lam_mid/2, 30))  # sicherer Abstand: halber Eigenwert
    # Rump-artige Verifikation: X = L̃ L̃ᵀ + c·I mit exakten Punkt-Einträgen ⇒ λ_min(X) ≥ c,
    # und λ_min(M) ≥ c − ‖M − X‖₂ ≥ c − ‖M − X‖_F  (rigoros in Ball-Arithmetik).
    c_mp = lam_mid*mp.mpf("0.99")
    Lc = mp.cholesky(Mm - c_mp*mp.eye(N))
    dig = int(prec*0.3)
    Lb = [[arb(mp.nstr(Lc[i, j], dig, strip_zeros=False)).mid() if j <= i else arb(0) for j in range(N)] for i in range(N)]
    Lm = arb_mat(N, N, [Lb[i][j] for i in range(N) for j in range(N)])
    cand = arb(mp.nstr(c_mp, dig)).mid()
    X = Lm*Lm.transpose()
    fro2 = arb(0)
    for i in range(N):
        for j in range(N):
            e = M[i, j] - X[i, j] - (cand if i == j else 0)
            u = abs(e).upper(); fro2 += u*u
    resid = fro2.sqrt()
    head_lower = cand - resid
    ok = bool(head_lower > 0)
    if verbose: print(f"Residuum ‖M−X‖_F ≤ {float(resid.upper()):.3e}", flush=True)
    res = dict(L=float(L.mid()), T_sharp=float(T.mid()), N=N, panels=panels, order=order, prec_bits=prec,
               A_L=str(A_L.mid())[:12], T1=float(T1.mid()), beta=float(beta.mid()),
               lambda_mid=mp.nstr(lam_mid, 8), cert_level=mp.nstr(lam_mid*mp.mpf("0.99"), 8), verified=ok,
               eps_B=float(eps_B.upper()), eps_D=float(eps_D.upper()),
               build_s=round(t_build, 1), total_s=round(time.time()-t0, 1))
    res["head_lower"] = float(head_lower.lower())
    res["quad_rigorous"] = quad_ok
    res["eps_quad"] = float(eps_quad.upper()) if quad_ok else None
    if ok:
        lower = head_lower - eps_B - (eps_quad if quad_ok else arb('inf'))
        res["certified_lower_bound"] = float(lower.lower())
    if verbose: print(json.dumps(res, indent=1), flush=True)
    return res, lam_mid

if __name__ == "__main__":
    L = float(sys.argv[1]); T = float(sys.argv[2]); N = int(sys.argv[3]); P = int(sys.argv[4])
    prec = int(sys.argv[5]) if len(sys.argv) > 5 else 320
    run(L, T, N, P, prec=prec)
