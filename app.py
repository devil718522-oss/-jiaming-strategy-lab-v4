import streamlit as st
import pandas as pd, numpy as np, requests
from datetime import date
from itertools import product

st.set_page_config(page_title="家銘策略實驗室 V4",page_icon="🧪",layout="wide")
st.title("🧪 家銘台股趨勢雷達 V4｜Walk-Forward 壓力測試")
st.caption("不追高勝率：用滾動樣本外、跨股票、市場狀態與大贏家敏感度檢驗策略。")
API="https://api.finmindtrade.com/api/v4/data"
DEFAULT="""2330
2303
2409
3481
2344
6770
2454
2382
3231
2317
2603
2609
2002
1301
1303
2881
2882
2886
2891
2884
2412
3711
3034
2379
2357
2356
2324
2376
3008
3443
5871
1216
1101
1102
2207
2201
2615
2618
2801
2912
6505
1326
2105
9910
1476
1590
2353
2354
2474
2610"""

@st.cache_data(ttl=3600,show_spinner=False)
def getp(sid,start,end,token):
    h={"Authorization":f"Bearer {token}"} if token else {}
    q={"dataset":"TaiwanStockPrice","data_id":sid,"start_date":str(start),"end_date":str(end)}
    r=requests.get(API,params=q,headers=h,timeout=30); r.raise_for_status()
    d=pd.DataFrame(r.json().get("data",[]))
    if d.empty:return d
    d["date"]=pd.to_datetime(d.date)
    return d.sort_values("date").rename(columns={"close":"Close","Trading_Volume":"Volume"}).reset_index(drop=True)

def prep(x):
    x=x.copy()
    for n in [5,10,20,60,200]:x[f"MA{n}"]=x.Close.rolling(n).mean()
    x["VR"]=x.Volume/x.Volume.rolling(20).mean()
    x["RET20"]=x.Close.pct_change(20); x["HH20"]=x.Close.rolling(20).max()
    return x.dropna().reset_index(drop=True)

def sig(x,p):
    vm,gap,mom,ema,tl=p
    tr=(x.Close>x.MA5)&(x.MA5>x.MA10)&(x.MA10>x.MA20)
    if tl==3:tr &= x.MA20>x.MA60
    en=tr&(x.MA5>x.MA5.shift())&(x.MA10>x.MA10.shift())&(x.VR>=vm)&((x.Close/x.MA5-1)<=gap)&(x.RET20>=mom)&(x.Close>=x.HH20.shift())
    return en.fillna(False),(x.Close<x[f"MA{ema}"]).fillna(False)

def bt(x,p,cost):
    en,ex=sig(x,p); out=[]; hold=False
    for i,r in x.iterrows():
        if not hold and en.iloc[i]: hold=True; ep=r.Close; ed=r.date
        elif hold and ex.iloc[i]:
            out.append((ed,r.date,r.Close/ep-1-cost)); hold=False
    if hold:out.append((ed,x.iloc[-1].date,x.iloc[-1].Close/ep-1-cost))
    return out

def met(rr):
    s=pd.Series(rr,dtype=float)
    if s.empty:return None
    win=s[s>0].sum(); loss=abs(s[s<=0].sum())
    pf=win/loss if loss else (5 if win else 0)
    eq=(1+s).cumprod()
    return dict(n=len(s),wr=(s>0).mean(),pf=pf,exp=s.mean(),total=eq.iloc[-1]-1,mdd=(eq/eq.cummax()-1).min(),median=s.median(),best=s.max())

with st.sidebar:
    train=st.slider("訓練視窗（年）",2,4,3)
    test=st.select_slider("每次樣本外測試（月）",[3,6,12],12)
    yrs=st.slider("總資料年數",5,10,7)
    cost=st.number_input("完整交易成本＋滑價（%）",0.,2.,0.55,.05)
    mino=st.slider("最低樣本外交易",30,300,80,10)
    token=st.text_input("FinMind Token",type="password")

stocks=[s.strip() for s in st.text_area("股票池",DEFAULT,height=250).splitlines() if s.strip()]
PARAMS=list(product([1.,1.2,1.5],[.03,.05,.08],[0.,.03,.06],[10,20],[2,3]))

if st.button("🔥 開始 V4 Walk-Forward 壓力測試",type="primary",use_container_width=True):
    end=pd.Timestamp(date.today()); start=end-pd.Timedelta(days=365*yrs+300)
    data={}; fail=[]; pg=st.progress(0)
    for i,s in enumerate(stocks):
        try:
            d=getp(s,start.date(),end.date(),token)
            if len(d)>500:data[s]=prep(d)
            else:fail.append(s)
        except:fail.append(s)
        pg.progress((i+1)/len(stocks),text=f"下載 {i+1}/{len(stocks)}")
    pg.empty()
    st.success(f"成功載入 {len(data)} 檔；失敗/不足 {len(fail)} 檔")
    if len(data)<10:st.error("資料不足；建議填 FinMind Token 後重試。");st.stop()

    common=max(x.date.min() for x in data.values()); ts=common+pd.DateOffset(years=train); wins=[]
    while ts<end:
        te=min(ts+pd.DateOffset(months=test),end); wins.append((ts,te)); ts=te

    rows=[]; periods=[]; bar=st.progress(0)
    for wi,(ts,te) in enumerate(wins):
        tr0=ts-pd.DateOffset(years=train); cand=[]
        for p in PARAMS:
            rr=[]
            for x in data.values():
                z=x[(x.date>=tr0)&(x.date<ts)].reset_index(drop=True)
                if len(z)>200:rr += [q[2] for q in bt(z,p,cost/100)]
            m=met(rr)
            if m and m["n"]>=20:
                score=min(m["pf"],3)*20+m["exp"]*200+min(m["n"],100)/10-abs(m["mdd"])*15
                cand.append((score,p))
        if not cand:continue
        p=max(cand,key=lambda q:q[0])[1]; wr=[]
        for sid,x in data.items():
            z=x[(x.date>=ts)&(x.date<te)].reset_index(drop=True)
            for ed,xd,r in bt(z,p,cost/100):
                base=x.iloc[(x.date-ed).abs().argsort()[:1]].iloc[0]
                regime="多頭" if base.Close>base.MA200 and base.MA60>base.MA200 else ("空頭" if base.Close<base.MA200 and base.MA60<base.MA200 else "盤整/轉折")
                rows.append(dict(stock=sid,entry=ed,exit=xd,ret=r,regime=regime,window=str(ts.date()),param=str(p)));wr.append(r)
        m=met(wr)
        if m:periods.append(dict(period=f"{ts.date()}→{te.date()}",**m,param=str(p)))
        bar.progress((wi+1)/len(wins),text=f"Walk-Forward {wi+1}/{len(wins)}")
    bar.empty()
    if not rows:st.error("沒有樣本外交易。");st.stop()
    df=pd.DataFrame(rows); m=met(df.ret)
    st.subheader("🏆 Walk-Forward 真正樣本外")
    cols=st.columns(4)
    for c,(k,v) in zip(cols,[("交易",m["n"]),("勝率",f'{m["wr"]:.1%}'),("PF",f'{m["pf"]:.2f}'),("每筆期望",f'{m["exp"]:.2%}')]):c.metric(k,v)
    cols=st.columns(4)
    for c,(k,v) in zip(cols,[("逐筆複利",f'{m["total"]:.1%}'),("最大回撤",f'{m["mdd"]:.1%}'),("中位數",f'{m["median"]:.2%}'),("最大單筆",f'{m["best"]:.1%}')]):c.metric(k,v)

    s=df.ret.sort_values(ascending=False); n=max(1,int(len(s)*.05)); tm=met(s.iloc[n:])
    st.subheader("🧨 移除最大 5% 贏家")
    a,b,c=st.columns(3);a.metric("PF",f'{tm["pf"]:.2f}');b.metric("期望",f'{tm["exp"]:.2%}');c.metric("移除筆數",n)

    st.subheader("🌦️ 市場狀態")
    rg=[]
    for name,g in df.groupby("regime"):
        q=met(g.ret);rg.append([name,q["n"],q["wr"],q["pf"],q["exp"],q["mdd"]])
    rdf=pd.DataFrame(rg,columns=["市場","交易","勝率","PF","期望","MDD"])
    st.dataframe(rdf.style.format({"勝率":"{:.1%}","PF":"{:.2f}","期望":"{:.2%}","MDD":"{:.1%}"}),use_container_width=True)

    w=pd.DataFrame(periods); pos=(w.exp>0).mean(); pfp=(w.pf>1).mean()
    st.subheader("📅 Walk-Forward 各期")
    st.dataframe(w.style.format({"wr":"{:.1%}","pf":"{:.2f}","exp":"{:.2%}","total":"{:.1%}","mdd":"{:.1%}"}),use_container_width=True)
    passed=m["n"]>=mino and m["pf"]>=1.25 and m["exp"]>0 and tm["pf"]>=1.05 and pos>=.6 and pfp>=.6
    st.subheader("🛡️ V4 判決")
    (st.success if passed else st.warning)(f'{"初步通過" if passed else "尚未通過"}｜正期望視窗 {pos:.0%}｜PF>1 視窗 {pfp:.0%}')
    st.caption("逐筆複利未處理同時持倉與資金配置，不可視為實際 CAGR。研究用途，不構成投資建議。")
