"""Candidate likelihood cost decomposition for the 50 x 100 experiment."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sparse_idcs import incremental_solver as solver
from sparse_idcs.embedded import decode_binary
from sparse_idcs.research import configure_sparse, sparse_operator


TI_PAYLOAD = """c-rk<J+kbo5f-I@kR`AMufS}{J3uZ6FZ7bE%%PWHxx4`5dcjj53(rUby5Zwz=A>_eTj$OZqG_7``MT*dP2}Ufo4)@TfBs;!@_ze%
wD5la!Q|E-KNu}PUbrAX&)zQ|4*wHd_@{fO*uq<+&*0^=OEdc#>HZNX|7-Su@M&g`cl<T!wHZ3S--hRGnEd+Xfk$7WAg|HqP3IN{
(T6;u`}(+NZI?*JHO`S29+U2$&w8)z{loOoaQ!wO6OV+|`22J8=y-nX2-{9{da}=P^T<8xC8IrN8vkrEEFr3e*{82Mq^Bjkxqc7|
OJUV?z(QZPC(>s;bJ%D9_pm2qxVuPS!)SO+9xC-Sod~PH81;JD5gt0R#*1sb9x#~zT$7vHF;Q_z=6c|@l)qhz5QKfzg-=NYugPBz
Zug@POWt3K>cUMDLG^y(wfsi|o5Vg_Iim0N5*2c7+VZ?H(JpJFCHejls%^3-mBfo8kF^p#J&FsJij(V_MrRt&b2V};eSN)7qPgXZ
N=>&?W*YE4d{h4Y+67stOl_EabLQ8=UK&n=9TVQ?8n4iR>Qb*W5>@N-gD;UHc<FWvNY~}TH%O-CElOB5x$M{>$pCvcmkWP?w?mO$
2Nqs8;%;Q&(|}a*JsU93%0tr6LB9mCLk@ct3bAM0CqbmEVOLIp1&VQ34v}g6G(7nRzFUVU-+w~7Jb%J<tkQqtz0W`L-lrdV?=Lz3
iTAF5#5eh9m-+Z^@O{F?n3h2r;?Yyb7d(HQB;~(|+-^I+N+$+&9CkQYTMq-;3;%0!)z(kN<u$}`zwsk+@tdHd&z4M^(je{8$MV>j
Ctk5DNI!#FQr_2spy-96NOOLMNy<CBgWYjC*B26i=r~;RMa&*fZZ1lr%yO^am2r?mn*Ox{^LXY33ok98@Z{<L*L$*vWxHQ^r0-qZ
H7LRqbBTe6lgo041*cv68Z+K=wZpY73DX@oxJ+7I&8x|R4Tr<wsYSc@x+TQWq=e~o?*yr8jD2Qt--?gv*3kjgg^R|iX7GbXdbh8W
LR6-J=CQtpO|b1ho`?dU`=zJ3x0Y7kd)o7C7?hqet-jxmOb#)n<ZJ)x`<BDiHpCycX9XqR*IOmZt_3$xqSei_?&JuhltI)~*Zq>^
u8U|t`tFx4XqQn?`;!+j7gdyf`<)#WLC?MFdp?GrCri$~7;qlA8q<_?J>Yh+n0Th7ivh<&u+zp=mKpGRSxBvDG(3-ZJ&bmy743k>
$btezb%Px7c1`XUB%X{!3sh!UT&71t6^6>w(eu#chCHPxtL#}j+$-5~a@gjRQMJ|_;uYgUc4;Fd;W({SL>2`0za)QYMj;e7zklV}
b6ZvIpAh3>GGflU+CQo7ivu_|)<A!E^0R{i3ayePKA7O1$jmf`LVOd{1GF5bVyad^i)8^^?vT2IeACw+llVtQ+PT*Sp%{5f2$@!g
J&@eKB6$;!eEGCy8<jDgoGX3%P-5?IXX0!;0lZ*xZ|Hxbp|g*8Ng`@8!SswlnULLn`cAffCmuHQycu+DN<UY7n%G&M-K50sD7|sg
-CvT1<zB2|Y8C~#Gf8RX5sM<Ei#$fkA-lqbo{10(9OLTvb099Ab4rMX3n$sm;OfpaskZ{Q5h=1nxjd_y)0>Zw$(2t|AzL^>8)fdR
z7O4eb3%kUv;0RcQ5!u`hYI>5`q!FGpos>{M!zPU!#-&2Lem&DPps%G^K+v)mB@t}EzO^Jz1eiYkOnCFGIh$g!H@<hDnsUuq%BKD
*%<vl--^7OP_*PD&k3x(4vNZjtkgAd2E%R5Lz^??VY$Cqx{*H>{UiLmmGXGWR1^5~;XnI?GTOGrH^uDawPaOnysE{&C?{fkc^sOg
AHajCcy3_NPT+tXO|XZNzCav-%uq!gPk?B!^E%g2hxtkd;ph>l?vnJSwxokdWHpA^sv)IIqpJdQ)FaY0j;@p52*#Iq8kSFUB7&yp
w;I=v7Vy75qF)!7wS_^q!DnY1kiP8<e$#by(hjY>Xfv51d!T!}luLEyp4#A<>5NwGJWLPUYbR1fC!?!6S)5KjxR*j|<BIA>*~3(1
_50`d&+ng)KkAzdFtp^Kf5Zs#K6T8=xA}kVe==OxeD`^zd`8xCNb^W$S=8c|xo04Yl-~xh@w&3sIr6iX7LG0b4`9(pq7#@``c4HF
ebQyd)f=Dt2KJB+xE>2*-(D>12;cnX!NNEG^i~99YkV3Y_xrt!U)by-
uREP!$y)?{4p6tc-7Gi&3QO7cK%x2Q6O&P}?ZILB?B@QJ!m+HP0xB4IcYXfxu+YHCdG1DCgwoOf4ZCIF<hDp)Y&89T*cc3=$qn-p
C4v0z-H7=Og*@WSl{DeEK@tQntIkGe2{%D14elp3si7^EnBhu6MZtVZ(^GgL!7<Kr!zNb<3D~&a_Z$B$%C~Jynzx4UXtT*hwzeFT
;Ruh;Cl{H5%xdiT+h$O<!dR1IGTOtSvI1X;%+m0*gaU?WrsUboB(MaaK(U}^2;!c-dSPt&&)EuKi(ob@+WB8OIfWzgU{>-hgEt5~
TWhh2$W-g!OIu0cui}=9<*IcD#x5bn1tz02)rw*lOHlBm0n2&+FW~x+XL1D5@+#~dRg9*zYbmV<6Su#hU^h*fkrpe<L*ctU5blud
@iL*K6>OPAF<%teQb%wjwOE{)x^r(L<9en&{RXQB=9W*iQ6mmP0)u^`Xgy`n(>Mp*GQQ+~$0Azk#LRVLw|VZL%+W+;s0K^b2C)zx
=Q3yH4ql+%t&SsIxN1Q0rC*BaTJCKNRFb=8YV-XFx$~>c@;-d=Fd5f#q}_vp%9{=Egm*mMm8yZ)qgM{%mn1s#BQ5XjXB3oP$%4&Y
>UXMN9aD31xQJlqUy}!R^{NI`5e>YKwh4o}6*h(M^<tb#q^;BF(t|H|#=Kg;$VKGGhT8DCJNb2P1Ud4WrZc-0Hdpdlvh=x6ukM0c
=DGgl(?+<8n==Bvks2@kRY|w2cxZ#saHQyqdh|58{kWo=7K?sWj~`9uclG$B1m%l+48i7!Zwl#?vtYj@eJR9#{9{rQ<@ko{M?5s2
lP(GATT8>&gni5Q?1NhVREHX`rBbsk+mM!KD8U_$`8>bBwKSG1P=}{5eIuDd$PGcOKbYOUvPy!O{H>y^ml6j$lD+n!n0>>fZ-vM3
qWv0u=-h&q9@0L^!dgC`|M>Q;imP7wGAzpXK4(!Xcb5nG++ncwH@s%tDc{yfyUa@z4FYKFcB!Y?Cq;Efr89>eEJXzACAv$rZUH%o
wlsD0Cw%#KKrY1RGAFbQ<avIzsrZU<V}Byf>xz=R>ECm^d&KPSmGrau9quiOV-^YNx5u?b88zX``SbaO{Cgb@m6r%CuN6ga)cUE!
#0VomGWGS+qvitVf>@S;6jdoN4oNh6pU}lrRL6M@|64)2K$wEV!}k)|1-2MoQhXoXxu|DR==Mpa4Afnx@=%oPXCt=d)1MDU_5OE#
!&j=(=U{;vb$wVGwRWz*uS(WRPW`R@P6qGf`4r=GFoyna$^FF=jk`Aon$R6M^9K^lz3DnqB@3g{6_LJy$u~6mKwRI!<U5Wl%6$W0
sY*5hq+(EU1zV%3r$|arrxg(a-u1kl_vn9D*WSCKtZs0{<4TFSL%PUInvai2w`o3s%B5U);Vbik2<QH&I(a&F<nz)h+w(C+nKRj$
dh3#YWdC2OZasRwtjqX#GDF+T1-e|qPP%Y6HBiJdVj{T*4M^A7K`AQ_!|PQpz2Y|PY|qx(DVCtJk1H~CA<NP-=^a31E8)^HRcL=}
&EJrLKdX6)n>A<9dk&&|Ct>1Ni-F6VeX6W=hJ6#(2^I2*Tob_7NjEqQ(g^t_O&lYuP>Q~no`HAC2f~rTZ3fGQix5t@rV5!Kkg}{|
b$3UVN+N-X++2<hPlqYzN>Uvs+{4qer?`eLQiT#5xWFdm2!BECord-;cHgiheIr550JP2#%CQj^Sz$%KtSo$G7?gZ~)q-||R2#qB
aIM99RHqF)1nHDzn94ZSXSa=Ze1wulopEER2^O@)Q8tw*SBB4^K*QYrx-&4o=Ljimgtf3PSt57)qLC5s;_%A)mKC}--9H8Qz;x<c
wh^wMT#M9#mpP!Id~|ko@c^m6bgl@spjJBKsv4LZ$7*2(@b{uIFZE_DWg*wd%A<T#oOKfe&&2et+*pSaY7bK~wryB0(58pF4wIy9
0jlwwVDJ)eF}b+#1*!ENZ+a%2w$fE8z#R_RLuyQSY!w{B!n1C0s)E%8-^A`{G%>|z#VXvrS@^>8rckEhmWIR1uh7Zz#?(@S7?<L-
WETZ%rZ~~gPm8^Kgsu7BPBka0D8AtUP`v>BEaF3xpXloSlwbjZANXdr$*q*G$51A?K^s%xE92wdNn<?wf*XP~x;#BPHnqk<0o*zo
cEV;P`&beY7wMZu45cvm#Yo>;i3ksC+S;gB1Af~(kXwiLt<>{lv#;P-D&NVRF`%Mw&)c!5<?cZd4XS=&KiTY}T4pdcquGVl^z~Ra
E@*@7zK$BqV%iwB4a*zw^E3Nw$4H-3E6lY}z$2uG^wSCamEjZ`HIc*FI9h_ZP%Sc={aZ^=cbN2~-wCKY50{eDXm%yCJj%DV1jNi}
CAgq>qz|Hew?V$;4i1Y(I%!DB?(D5O9*}nsm3vi@nb?hkWY6ENV`+ocyx4L<=3?bvY)@OaxwzQrkjiy@^m+g8Ec(*Xx+#-|MC5G>
qt3LZL1d)kG~PN@5In_6*Yys<$viB8c2|k=xcSP%T!q<|WnWk?YM}2A5`kA<fiO7cfy$7r-DwGQYCSp^c36|qPb=$7X_GBW^fhnl
y%+xH3s`bN;hFnw&qp<{VE8LXw5U7NdIW`}Iybs$XyDIVzU>iJiXCXp_ni8dabm53?8)4Sp3c`^@qWIJ(!Mqo=kh*?SwCOez1lS=
zhJ0;@=MRh?U1W(`>;C|&4nU5<F<DdM|qH33&>(0Cl4(ISC|j!87_4PRD69TkQiygncwjI@ZMBL;fKxp^T*9+l~SfC%ilhtJvStu
mHeClEBpHMU8NOQz~8@Mpvzt-PxXW0Ls!$9iK=f}u5$H%{ETz1$DlUn)((GMbLnS~yX_Ej`{aC=FX-)STWZe^aNs&ZzqJF<Zj{-K
eqB%O4(JF=KQK8;-^$SWq(2@=&CGkQ9zAUV-
WXcfbQ1cugfjS5xuzlo-7OUJaN9|B1;DH9qT<$3RXd#9L30QMW2=vDcU#C(Q?~Hxn)Y5?1%-PE?n63*^dD`?%sUNg@0Lc6Bts-cZ
jaW&H$iqNs4Yz!d%4EKx5%nB>3eGa>ss*6O9yVHo;S1)i}Llqv_P+_5Ps8AA=~OIycA&E!a*g}MRlg2oh?h?cqF&h5U{-j5|514!
0#~pFUCYt)RXr<n+jf0m;EU+y-2xgVRW7ZTewD;vq<v!OD?GwH9tOaZ#GW)&PY7t={YJ|XJtEna$asz`R7GAzPr*uGP}H<uE;u+_
y~N)e>CgSq_)y%DXq!i=72}!X%Cx*XMPPEx=uCPt1gI2D$&)^;hLq4KwhDb#R%y-R%P!{i>_(+H)Lbl%>u}s?H%&1kL=b*cW$Z}(
(ZBTM<wvrbJcU}T(Y?n<+k47yzLFz2h*r%ded`7H_{ukLR-uDE|aP|tR23Ir=R^%dwDO)a1Do1`tDu2P1lj5UgS789YWoUeRK=8>
R##+%_zb2V&*31HFu*v)4H}|yAgM5bic#X+z75jKVa2$pJwFm!`N20wR=CZJQ`~-Y>#%#TBt8EhWtpJkQ9RfJjOERAnc@3Hh1_Bw
%`jT!@Ytmz{BM*bRmJs?T5&9gTIV~$S6L0n6HNam>1!CybST{IA|luPV27W2Ya^T9azTpY{zltsH`JRJc<@!$LSarwsYXShNv#Xf
JZOPt^2pN(fh|njvkZq3uR4C_)@CdeVlFXQ@^HgEkE;Xm}qU45Mb<t@1NUe5}|+L*Gy9ja~ULE<K5AteB1Gg_CdZ?<Y<U*l@gv*G
xC2=PGMX"""
TI_SHA256 = "8f9420d9fad0ae73f7ce8d8cbb4bd8280d364758aa54594aeb3e23ead8bd7875"
TRUTH_PAYLOAD = """c-pm4%Lzj<40CyhcgzS~q>$0_Ht8x4x=2O{^w1naR7*)emqIBt2L)S3mTdOFcW6QoSM0*#7|?`J2bD9xW(!yl!A_01ItJER-E0G+
21#_AL{e*oKJrgJQeEq~+_8<dvL;tit`-sg$-4+UmRF8n-X=T=gA|vcHI~FK!tHZ!6(J_hJITdx93SW6aAG3n%VPQ%&KjP~uCbYQ
52#*v`|L8j4vLP{bzNM94WAx0ZZ4Tn<jlf*jFHUBS0}jx%<0;O"""
TRUTH_SHA256 = "7ebb1bd5372813e1f6ac8a59c04d56314ac6f6803731b89ae509bd209efb4968"
ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=int, default=400)
    parser.add_argument("--smoke", action="store_true", help="20 x 30 cells, 10 evaluations")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    ti = decode_binary(TI_PAYLOAD, (500, 500), TI_SHA256)
    truth = decode_binary(TRUTH_PAYLOAD, (100, 50), TRUTH_SHA256)
    if args.smoke:
        truth = truth[:30, :20]
    configure_sparse(truth.shape, 20260829)
    a, csc, observed, _, _, _, _ = sparse_operator(truth)
    mean_s = float(np.mean(solver.slowness_from_binary(ti)))
    model = np.full(truth.size, mean_s)
    base_residual = observed - a @ model
    base_loglik = -0.5 * float(np.dot(base_residual, base_residual))
    dense = a.toarray()
    active = np.flatnonzero(np.diff(csc.indptr) > 0)
    rng = np.random.default_rng(20260908)
    count = 10 if args.smoke else args.evaluations
    if count <= 0:
        raise ValueError("evaluations must be positive")
    cells = rng.choice(active, size=count, replace=True)
    values = np.where(rng.integers(0, 2, size=count) > 0,
                      1.0 / 0.06, 1.0 / 0.08)

    def dense_full():
        gains = np.empty(count)
        for k, (cell, value) in enumerate(zip(cells, values)):
            candidate = model.copy()
            candidate[int(cell)] = value
            residual = observed - dense @ candidate
            gains[k] = -0.5 * float(np.dot(residual, residual)) - base_loglik
        return gains

    def csr_full():
        gains = np.empty(count)
        for k, (cell, value) in enumerate(zip(cells, values)):
            candidate = model.copy()
            candidate[int(cell)] = value
            residual = observed - a @ candidate
            gains[k] = -0.5 * float(np.dot(residual, residual)) - base_loglik
        return gains

    def csc_incremental():
        gains = np.empty(count)
        for k, (cell, value) in enumerate(zip(cells, values)):
            indices, weights = solver.sparse_column(csc, int(cell))
            gains[k] = solver.candidate_data_loglik(
                base_residual, indices, weights, float(value - model[int(cell)])
            )
        return gains

    measured = []
    for name, function in (
        ("dense_full", dense_full), ("csr_full", csr_full), ("csc_incremental", csc_incremental)
    ):
        t0 = time.perf_counter()
        gains = function()
        elapsed = time.perf_counter() - t0
        measured.append((name, elapsed, gains))
    reference = measured[0][2]
    for name, _, gains in measured[1:]:
        if not np.allclose(gains, reference, atol=1e-8, rtol=1e-8):
            raise AssertionError(f"Candidate gains differ for {name}")
    rows = [{
        "method": name, "evaluations": count, "total_seconds": elapsed,
        "microseconds_per_evaluation": elapsed / count * 1e6,
        "max_abs_gain_error": float(np.max(np.abs(gains - reference))),
    } for name, elapsed, gains in measured]
    out = args.output or ROOT / "run_outputs" / "likelihood_cost" / ("smoke" if args.smoke else "paper")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "cost.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.bar([row["method"] for row in rows],
           [row["microseconds_per_evaluation"] for row in rows], color="#557f9a")
    ax.set_yscale("log")
    ax.set_ylabel("Microseconds per candidate")
    ax.tick_params(axis="x", rotation=15)
    fig.savefig(out / "candidate_cost.png", dpi=180)
    plt.close(fig)
    print(json.dumps(rows, indent=2), flush=True)
    print(f"Generated outputs: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
