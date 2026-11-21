import pandas as pd,itertools,json,networkx as nx
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
df=pd.read_csv("daddy_yankee_songs.csv",encoding="utf-8-sig")
m=dict(l.rstrip("\n").split("|") for l in open("collabs.txt",encoding="utf8") if l.strip())
miss=[t for t in m if t not in set(df.title)];print("unmatched:",miss)
E=[(t,a) for t,s in m.items() for a in s.split(";")]
ed=pd.DataFrame(E,columns=["title","collaborator"]).merge(df[["title","year","latin","hot100"]],on="title")
ed.to_csv("daddy_yankee_collaborations.csv",index=False,encoding="utf-8-sig")
g=ed.groupby("collaborator").agg(songs=("title","count"),first=("year","min"),last=("year","max"),best_latin=("latin","min")).reset_index().sort_values(["songs","best_latin"],ascending=[False,True])
g.to_csv("collaborator_summary.csv",index=False,encoding="utf-8-sig")
G=nx.Graph()
for t,s in m.items():
    for a,b in itertools.combinations(s.split(";"),2): G.add_edge(a,b)
dc=nx.degree_centrality(G);top=sorted(dc,key=dc.get,reverse=True)[:5]
print(g.head(10).to_string());print("collaborators",len(g),"songs",len(m),"top centrality",top)
f,ax=plt.subplots(1,2,figsize=(7.2,3.4),gridspec_kw={"width_ratios":[1,1.15]})
t=g.head(10).iloc[::-1];ax[0].barh(t.collaborator,t.songs,color="#e8a200");ax[0].set_xlabel("Songs with Daddy Yankee (in dataset)");ax[0].spines[["top","right"]].set_visible(False);ax[0].tick_params(labelsize=7)
H=G.subgraph([n for n in G if G.degree(n)>=3]);pos=nx.spring_layout(H,seed=4,k=.9)
nx.draw_networkx_edges(H,pos,ax=ax[1],alpha=.25,width=.6);nx.draw_networkx_nodes(H,pos,ax=ax[1],node_size=[60+40*g.set_index("collaborator").songs.get(n,1) for n in H],node_color="#c2410c",alpha=.85)
nx.draw_networkx_labels(H,pos,ax=ax[1],font_size=5);ax[1].axis("off");ax[1].set_title("Co-appearance network (3+ links)",fontsize=8)
f.tight_layout();f.savefig("figs/f6_network.png",dpi=200)
json.dump({"top":g.head(12).fillna("").to_dict("records"),"n_collab":len(g),"n_songs":len(m),"central":top},open("net.json","w"),ensure_ascii=False,default=float)
