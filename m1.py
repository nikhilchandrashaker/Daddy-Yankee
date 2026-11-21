import numpy as np, pandas as pd, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows=[l.rstrip("\n").split("|") for l in open("songs.txt",encoding="utf8") if l.strip()]
df=pd.DataFrame(rows,columns=["title","year","role","album","hot100","latin","cert"])
df["year"]=df.year.astype(int)
for c in ["hot100","latin"]: df[c]=pd.to_numeric(df[c].replace("-",np.nan))
rn={"L":"Solo lead","C":"Lead with collaborators","F":"Featured artist"}
df["role_name"]=df.role.map(rn)
def era(y): return "Breakout" if y<=2003 else "Global" if y<=2010 else "Collab" if y<=2016 else "Streaming" if y<=2019 else "Farewell" if y<=2022 else "Post-retirement"
df["era"]=df.year.map(era)
df["top10"]=(df.latin<=10).astype(int); df["no1"]=(df.latin==1).astype(int)
df.to_csv("daddy_yankee_songs.csv",index=False,encoding="utf-8-sig")
alb=[(1995,"No Mercy","Studio",None,None),(1997,"El Cartel de Yankee","Compilation",None,None),(2001,"El Cartel II: Los Cangris","Compilation",None,None),(2002,"El Cangri.com","Studio",None,43),(2003,"Los Homerun-es","Compilation",158,7),(2004,"Barrio Fino","Studio",26,1),(2005,"Ahora le Toca al Cangri! Live","Live",104,3),(2005,"Barrio Fino en Directo","Live",24,1),(2007,"El Cartel: The Big Boss","Studio",9,1),(2008,"Talento de Barrio","Soundtrack",13,1),(2010,"Mundial","Studio",29,1),(2012,"Prestige","Studio",39,1),(2013,"King Daddy","Mixtape",None,7),(2019,"Con Calma & Mis Grandes Éxitos","Compilation",None,None),(2022,"Legendaddy","Studio",8,1),(2025,"Lamento en Baile","Studio",None,None)]
pd.DataFrame(alb,columns=["year","title","type","us200","top_latin_albums"]).to_csv("daddy_yankee_albums.csv",index=False,encoding="utf-8-sig")
R={}
c=df.dropna(subset=["latin"]).copy()
R["n_songs"]=len(df);R["n_latin"]=len(c);R["n_h100"]=int(df.hot100.notna().sum());R["n_no1"]=int(df.no1.sum());R["n_top10"]=int(df.top10.sum())
# OLS: Latin peak ~ year + collab + featured (charted songs)
X=np.column_stack([np.ones(len(c)),c.year-2004,(c.role=="C").astype(int),(c.role=="F").astype(int)]);y=c.latin.values.astype(float)
b,_,_,_=np.linalg.lstsq(X,y,rcond=None);res=y-X@b;s2=res@res/(len(y)-X.shape[1]);se=np.sqrt(np.diag(s2*np.linalg.inv(X.T@X)))
r2=1-res@res/((y-y.mean())@(y-y.mean()))
R["ols"]={"names":["Intercept","Year (since 2004)","Collab lead","Featured"],"b":b.round(3).tolist(),"se":se.round(3).tolist(),"r2":round(r2,3),"n":len(y)}
# Logistic: P(top10 on Hot Latin Songs) ~ year + collab + featured, all songs (uncharted=0), Newton-Raphson, L2 tiny
Xa=np.column_stack([np.ones(len(df)),df.year-2004,(df.role=="C").astype(int),(df.role=="F").astype(int)]);ya=df.top10.values
w=np.zeros(4)
for _ in range(50):
    p=1/(1+np.exp(-Xa@w));g=Xa.T@(ya-p)-0.01*w;H=-(Xa.T*(p*(1-p)))@Xa-0.01*np.eye(4);w=w-np.linalg.solve(H,g)
p=1/(1+np.exp(-Xa@w));cov=np.linalg.inv((Xa.T*(p*(1-p)))@Xa+0.01*np.eye(4));sel=np.sqrt(np.diag(cov))
R["logit"]={"names":["Intercept","Year (since 2004)","Collab lead","Featured"],"b":w.round(3).tolist(),"se":sel.round(3).tolist(),"or":np.exp(w).round(3).tolist(),"n":len(df)}
# accuracy (in-sample) vs base rate
R["logit"]["acc"]=round(float(((p>.5)==ya).mean()),3);R["logit"]["base"]=round(float(1-ya.mean()),3)
# era / role summaries
g=c.groupby("era").agg(n=("title","count"),median_peak=("latin","median"),no1=("no1","sum"),top10=("top10","sum")).reindex(["Breakout","Global","Collab","Streaming","Farewell","Post-retirement"]).dropna()
R["era"]=g.reset_index().to_dict("records")
r=c.groupby("role_name").agg(n=("title","count"),median_peak=("latin","median"),no1=("no1","sum"),top10=("top10","sum")).reset_index()
R["role"]=r.to_dict("records")
json.dump(R,open("results.json","w"),ensure_ascii=False,indent=1,default=float)
# FIGURES
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
A,B="#e8a200","#c2410c"
f,ax=plt.subplots(figsize=(6.5,3.2))
for k,col,m in [("L",A,"o"),("C",B,"s"),("F","#4b5563","^")]:
    d=c[c.role==k];ax.scatter(d.year+np.random.RandomState(1).uniform(-.2,.2,len(d)),d.latin,c=col,marker=m,s=28,label=rn[k],alpha=.85)
xs=np.array([2002,2025]);ax.plot(xs,b[0]+b[1]*(xs-2004),"k--",lw=1,label="OLS trend (solo)")
ax.invert_yaxis();ax.set_ylabel("Peak, Hot Latin Songs (1 = best)");ax.set_xlabel("Release year");ax.legend(frameon=False,fontsize=7);f.tight_layout();f.savefig("figs/f1_scatter.png",dpi=200);plt.close()
f,ax=plt.subplots(figsize=(6.5,2.8));gg=g.reset_index();x=np.arange(len(gg));ax.bar(x-.2,gg.top10,.4,color=A,label="Top-10 entries");ax.bar(x+.2,gg.no1,.4,color=B,label="No. 1 entries");ax.set_xticks(x);ax.set_xticklabels(gg.era);ax.legend(frameon=False);ax.set_ylabel("Songs (in dataset)");f.tight_layout();f.savefig("figs/f2_era.png",dpi=200);plt.close()
f,ax=plt.subplots(figsize=(6.5,2.8));rr=r.set_index("role_name").reindex(list(rn.values()));ax.bar(rr.index,rr.top10/rr.n*100,color=[A,B,"#4b5563"]);ax.set_ylabel("% of charted songs in top 10");[ax.text(i,v+1,f"n={n}",ha="center",fontsize=8) for i,(v,n) in enumerate(zip(rr.top10/rr.n*100,rr.n))];f.tight_layout();f.savefig("figs/f3_role.png",dpi=200);plt.close()
f,ax=plt.subplots(figsize=(6.5,2.4));lab=w[1:];err=1.96*sel[1:];ax.errorbar(lab,[2,1,0],xerr=err,fmt="o",color=B,capsize=3);ax.axvline(0,color="k",lw=.8);ax.set_yticks([2,1,0]);ax.set_yticklabels(["Year (since 2004)","Collab lead","Featured"]);ax.set_xlabel("Logit coefficient (95% interval), outcome = Latin top 10");f.tight_layout();f.savefig("figs/f4_logit.png",dpi=200);plt.close()
f,ax=plt.subplots(figsize=(6.5,2.0));ax.axis("off");E=[("1992-2001","Underground\nPlayero, No Mercy"),("2002-2003","Breakout\nEl Cangri.com"),("2004-2010","Global\nBarrio Fino"),("2011-2016","Collab\nPrestige, Shaky"),("2017-2019","Streaming\nDespacito, Con Calma"),("2020-2022","Farewell\nLegendaddy"),("2023-2025","Post-retirement\nChristian releases")]
for i,(a_,t) in enumerate(E):
    ax.add_patch(plt.Rectangle((i*1.0+.02,.3),.94,.55,fc=A if i%2==0 else "#f59e0b",ec="k",lw=.6));ax.text(i+.49,.57,t,ha="center",va="center",fontsize=6.3);ax.text(i+.49,.12,a_,ha="center",fontsize=7)
    if i<6: ax.annotate("",xy=(i+1.02,.57),xytext=(i+.97,.57),arrowprops=dict(arrowstyle="->"))
ax.set_xlim(0,7);ax.set_ylim(0,1);f.tight_layout();f.savefig("figs/f5_eras.png",dpi=200);plt.close()
print(json.dumps(R,ensure_ascii=False,indent=1,default=float))
