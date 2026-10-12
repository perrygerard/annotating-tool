import sys,re,math,collections,pickle
sys.path.insert(0,'/home/claude/annotation-remapper')
import pymupdf as fitz, derivative as D, remapper as R
src="/mnt/user-data/uploads/DX_deriv_lead-compressed.pdf"; der="/root/.claude/uploads/f79acecc-2547-5999-85bd-b4296fe131f3/b60c998a-DX_deriv_supportive.pdf"
gtf="/root/.claude/uploads/f79acecc-2547-5999-85bd-b4296fe131f3/bf99aaa6-DX_deriv_support_annos.pdf"
res=D.process_derivative(src,der,"/tmp/dx_out.pdf",None)
app=res["_all_annots"]
g=fitz.open(gtf); gt=[]
for a in g[0].annots():
    if a.type[0]==2:
        cl=g.xref_get_key(a.xref,"CL")[1]; nums=[float(n) for n in re.findall(r"-?\d+(?:\.\d+)?",cl)]
        ph=g[0].mediabox.height
        tip=(nums[0],ph-nums[1]) if len(nums)>=4 else None
        gt.append(dict(content=a.info.get("content",""),rect=fitz.Rect(a.rect),tip=tip))
norm=lambda s:re.sub(r"\s+"," ",s).strip().lower()
bycontent=collections.defaultdict(list)
for i in app: bycontent[norm(i.content)].append(i)
print("GT callouts",len(gt),"app",len(app))
out=[]
for k,gx in enumerate(gt):
    m=bycontent.get(norm(gx["content"]))
    if not m:
        # prefix match
        m=[i for i in app if norm(i.content)[:40] and norm(i.content)[:40]==norm(gx["content"])[:40]]
    if not m: print(k,"NO APP MATCH",repr(gx["content"][:50])); continue
    i=m[0]
    tip=None
    if i.new_vertices: tip=i.new_vertices[0]
    d=math.hypot(tip[0]-gx["tip"][0],tip[1]-gx["tip"][1]) if tip and gx["tip"] else None
    print(k,repr(gx["content"][:42]),i.status,i.match_method,round(i.match_confidence,2),"srcpage",i.page_num+1,"dist",None if d is None else round(d),"gt_tip",None if not gx["tip"] else tuple(round(x) for x in gx["tip"]),"app_tip",tuple(round(x) for x in tip) if tip else None)

print("\n=== fuzzy by URL / alt / REF ===")
def key(t):
    t=t.replace(" "," ")
    m=re.search(r"REF-(\d+)",t)
    if m: return "ref"+m.group(1)
    m=re.search(r"(https?://[^\s]+|www\.[^\s]+)",t)
    if m: return "url:"+m.group(1).rstrip("/").lower()
    m=re.search(r"<alt tag:\s*([^>]*)",t,re.I)
    if m: return "alt:"+norm(m.group(1))[:40]
    return None
appk=collections.defaultdict(list)
for i in app:
    k=key(i.content)
    if k: appk[k].append(i)
good=0;tot=0;rows=[]
for k,gx in enumerate(gt):
    kk=key(gx["content"]); m=appk.get(kk) or []
    if not m:
        # prefix-compare alt/url
        m=[i for kk2,l in appk.items() for i in l if kk and (kk2.startswith(kk[:30]) or kk.startswith(kk2[:30]))]
    if not m: print(k,"no source match",kk); continue
    best=None
    for i in m:
        tip=i.new_vertices[0] if i.new_vertices else None
        d=math.hypot(tip[0]-gx["tip"][0],tip[1]-gx["tip"][1]) if tip and gx["tip"] else 9999
        if best is None or d<best[0]: best=(d,i,tip)
    d,i,tip=best; tot+=1
    ok=d<80 and i.status!="content_removed"
    good+=ok
    print(k,kk[:50],"|",i.status,round(i.match_confidence,2),"srcpg",i.page_num+1,"n_src",len(m),"dist",round(d),"OK" if ok else "")
print("within 80pt & placed:",good,"of",tot)

print("\n=== precision: app-placed vs kept ===")
gtkeys={key(g_["content"]) for g_ in gt if key(g_["content"])}
def kept(i):
    k=key(i.content)
    return bool(k) and any(k==x or k.startswith(x[:30]) or x.startswith(k[:30]) for x in gtkeys)
placed=[i for i in app if i.status!="content_removed"]
print("placed",len(placed),"of which kept by copywriter",sum(kept(i) for i in placed))
for i in placed:
    if not kept(i): print("  not kept:",i.status,round(i.match_confidence,2),"srcpg",i.page_num+1,repr(i.content[:60]))
dropped=[i for i in app if i.status=="content_removed"]
print("dropped",len(dropped),"of which copywriter kept",sum(kept(i) for i in dropped))

print("\n=== exact-key precision ===")
def kept2(i):
    k=key(i.content); return bool(k) and k in gtkeys
print("placed",len(placed),"kept(exact)",sum(kept2(i) for i in placed))
print("dropped",len(dropped),"kept(exact)",sum(kept2(i) for i in dropped), [ (key(i.content) or "")[:45] for i in dropped if kept2(i)])
print("gt keys",len(gtkeys))

print("\n=== dup groups among placed ===")
pk=collections.defaultdict(list)
for i in placed:
    k=key(i.content)
    pk[k].append(i)
for k,l in pk.items():
    if len(l)>1 and k:
        print(k[:50],[(i.page_num+1,i.status,round(i.match_confidence,2),tuple(round(x) for x in i.new_vertices[0])) for i in l])
print("all placed:")
for i in placed: print(" ",i.page_num+1,i.status,round(i.match_confidence,2),tuple(round(x) for x in i.new_vertices[0]),(key(i.content) or i.content[:40])[:55])
