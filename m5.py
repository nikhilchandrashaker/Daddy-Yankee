import pandas as pd,numpy as np,re,json
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
df=pd.read_csv("daddy_yankee_songs.csv",encoding="utf-8-sig");al=pd.read_csv("daddy_yankee_albums.csv")
sales={"Barrio Fino":1083000,"Barrio Fino en Directo":809000,"El Cartel: The Big Boss":300000,"Talento de Barrio":200000,"Mundial":57842,"Prestige":50000,"King Daddy":2000}
cert={"Barrio Fino":"RIAA Platinum","Barrio Fino en Directo":"RIAA Gold","El Cartel: The Big Boss":"RIAA 3x Platinum (Latin)","Talento de Barrio":"RIAA 2x Platinum (Latin)","Prestige":"RIAA 2x Platinum (Latin)","Legendaddy":"RIAA Platinum (Latin)"}
al["us_sales"]=al.title.map(sales);al["cert"]=al.title.map(cert).fillna("");al.to_csv("daddy_yankee_albums.csv",index=False,encoding="utf-8-sig")
# robustness
c=df.dropna(subset=["latin"]);rs=stats.spearmanr(c.year,c.latin)
rng=np.random.RandomState(7);sl=[]
for _ in range(3000):
    s=c.sample(len(c),replace=True,random_state=rng);sl.append(np.polyfit(s.year,s.latin,1)[0])
ci=np.percentile(sl,[2.5,97.5])
grp=[g.latin.values for e,g in c.groupby("era") if e!="Breakout"];kw=stats.kruskal(*grp)
# no-2022 check: drop the 2022 Legendaddy-era burst to test sensitivity
c2=c[c.year!=2022];rs2=stats.spearmanr(c2.year,c2.latin)
h=df.dropna(subset=["hot100"]).groupby("era").size().to_dict()
R2={"rho":round(rs.statistic,3),"p":round(rs.pvalue,4),"boot_lo":round(ci[0],3),"boot_hi":round(ci[1],3),"kw_h":round(kw.statistic,2),"kw_p":round(kw.pvalue,4),"rho_no22":round(rs2.statistic,3),"p_no22":round(rs2.pvalue,4),"n":len(c),"n_no22":len(c2),"h100_by_era":h,"n_h100":int(df.hot100.notna().sum()),"min_h100":int(df.hot100.min())}
# certs
rows=[]
for r in df.itertuples():
    m=re.search(r"RIAA (\d+)x",(r.cert if isinstance(r.cert,str) else ""))
    if m: rows.append((r.title,int(m.group(1)),"Latin" in r.cert))
top=sorted(rows,key=lambda x:-x[1])[:10];R2["certs"]=[[t,n,l] for t,n,l in top]
json.dump(R2,open("results2.json","w"),ensure_ascii=False,indent=1)
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False});A,B="#e8a200","#c2410c"
a=al.dropna(subset=["us_sales"]).sort_values("year");f,ax=plt.subplots(figsize=(6.5,2.8));ax.bar([f"{t}\n({y})" for t,y in zip(a.title.str.replace("El Cartel: The Big Boss","El Cartel"),a.year)],a.us_sales/1000,color=A);ax.set_ylabel("US sales (thousands)");ax.tick_params(axis="x",labelsize=6.5);f.tight_layout();f.savefig("figs/f7_sales.png",dpi=200);plt.close()
f,ax=plt.subplots(figsize=(6.5,2.8));t=top[::-1];ax.barh([x[0][:34] for x in t],[x[1] for x in t],color=B);ax.set_xlabel("RIAA certification multiplier (Latin thresholds differ from standard)");ax.tick_params(axis="y",labelsize=7);f.tight_layout();f.savefig("figs/f8_certs.png",dpi=200);plt.close()
print(json.dumps(R2,ensure_ascii=False))
