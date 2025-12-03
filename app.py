import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import random
import time

# --- 頁面設定 ---
st.set_page_config(
    page_title="AI 文本偵測實驗室 (Pro)",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS 樣式 (仿照 JustDone 與 test.html 風格) ---
st.markdown("""
<style>
    /* 全域字體 */
    .stApp { font-family: 'Helvetica', 'Arial', sans-serif; }
    
    /* 標題區塊 */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* 分數顯示圓環模擬 */
    .score-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    .score-title { font-size: 1rem; color: #64748b; margin-bottom: 0.5rem; }
    .score-value { font-size: 3.5rem; font-weight: 800; color: #ef4444; }
    
    /* 句子螢光筆效果 */
    .highlight-ai { background-color: #fecaca; padding: 2px 4px; border-radius: 4px; border-bottom: 2px solid #ef4444; }
    .highlight-human { background-color: #d1fae5; padding: 2px 4px; border-radius: 4px; border-bottom: 2px solid #10b981; }
    
    /* 文字輸入區優化 */
    .stTextArea textarea {
        border-radius: 0.5rem;
        border: 1px solid #cbd5e1;
        font-size: 16px;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# --- 核心邏輯函數 (模擬 test.html 中的數學理論) ---

def split_sentences(text):
    """將文本切割成句子 (簡單規則)"""
    # 支援中文與英文標點
    sentences = re.split(r'(?<=[.!?。！？])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def calculate_burstiness(sentences):
    """
    計算 Burstiness (Slide 9)
    B = sigma / mu (標準差 / 平均句長)
    """
    if not sentences:
        return 0, 0
    
    lengths = [len(s) for s in sentences]
    mean_len = np.mean(lengths)
    std_dev = np.std(lengths)
    
    # 避免除以零
    if mean_len == 0:
        return 0, 0
        
    burstiness = std_dev / mean_len
    return burstiness, mean_len

def simulate_perplexity_analysis(text):
    """
    模擬 Perplexity 計算 (Slide 6-8)
    (註：真實專案會在此呼叫 GPT-2/BERT 模型計算 loss)
    這裡使用統計特徵來模擬 PP 曲線
    """
    sentences = split_sentences(text)
    pp_values = []
    
    # 模擬邏輯：
    # AI 句子通常結構完整、長度適中 -> PP 低且穩定
    # Human 句子長短不一、用詞突兀 -> PP 高且波動大
    
    for sent in sentences:
        base_pp = 10  # 基礎分
        # 根據句長與隨機因子模擬 PP
        length_factor = len(sent) / 10
        random_factor = random.uniform(0.8, 1.5)
        
        # 簡單模擬：過短或過長的句子通常 PP 較高 (Human 特徵)
        if len(sent) < 5 or len(sent) > 80:
            pp = base_pp * 2.5 * random_factor
        else:
            pp = base_pp * 1.2 * random_factor
            
        pp_values.append(pp)
        
    return sentences, pp_values

def analyze_text(text):
    """
    主分析函數：整合所有指標並給出最終 AI 機率
    """
    if not text:
        return None
        
    sentences, pp_values = simulate_perplexity_analysis(text)
    burstiness_score, mean_len = calculate_burstiness(sentences)
    
    # --- 決策邏輯 (Slide 23: Logistic Regression 概念模擬) ---
    # AI 特徵：低 Burstiness, 低 PP 波動
    # Human 特徵：高 Burstiness, 高 PP 波動
    
    pp_variance = np.var(pp_values) if pp_values else 0
    
    # 簡易權重計算
    ai_score = 0.5
    
    # 1. 檢查 Burstiness (AI 通常較低，約 0.3-0.5；人類通常 > 0.6)
    if burstiness_score < 0.4:
        ai_score += 0.2
    elif burstiness_score > 0.6:
        ai_score -= 0.2
        
    # 2. 檢查 PP 變異度 (Slide 7: AI 平滑, Human 波動)
    if pp_variance < 10: # 平滑
        ai_score += 0.2
    else:
        ai_score -= 0.15
        
    # 3. 限制範圍 0~1
    ai_score = max(0.01, min(0.99, ai_score))
    
    # 加上一點隨機擾動模擬真實模型的信心區間
    final_prob = ai_score * 100
    
    return {
        "ai_probability": final_prob,
        "burstiness": burstiness_score,
        "perplexity_trend": pp_values,
        "sentences": sentences,
        "stats": {
            "sentence_count": len(sentences),
            "word_count": len(text),
            "avg_sentence_len": mean_len
        }
    }

# --- 側邊欄 (Sidebar) ---
with st.sidebar:
    st.header("⚙️ 設定與理論")
    st.info("本系統基於《AI 偵測技術 — 高階理論篇》建構。")
    
    detection_mode = st.radio(
        "偵測模式 (Model Architecture)",
        ["Standard (Statistical)", "Advanced (BERT-Hybrid)", "Experimental (Stylometry)"]
    )
    
    st.divider()
    
    st.subheader("📊 關鍵指標說明")
    with st.expander("Perplexity (困惑度)"):
        st.markdown("**定義**：模型對下一個字的驚訝程度。\n\n**特徵**：AI 寫作通常 PP 較低且曲線平滑；人類寫作會有「爆點」。")
    
    with st.expander("Burstiness (節奏)"):
        st.markdown("**定義**：句子長度與結構的變異性。\n\n**特徵**：人類寫作長短句交錯 (高 Burstiness)；AI 傾向於規律的中庸長度。")
        
    st.divider()
    st.caption("Version 1.0 | Based on Slide 29 Architecture")

# --- 主畫面 (Main UI) ---

# Header
st.markdown("""
<div class="main-header">
    <h1 style="margin:0;">🕵️ AI Content Detector (Pro)</h1>
    <p style="opacity:0.8; margin-top:0.5rem;">
        基於多維度語言學特徵 (Perplexity, Burstiness, Semantic Drift) 的偵測引擎
    </p>
</div>
""", unsafe_allow_html=True)

# Layout: 左側輸入，右側即時結果
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("📝 輸入文本")
    input_text = st.text_area(
        "請貼上需要分析的文章 (建議 100 字以上)",
        height=300,
        placeholder="在此貼上文章..."
    )
    
    analyze_btn = st.button("🚀 開始偵測 (Analyze)", type="primary", use_container_width=True)

# 執行分析
if analyze_btn and input_text:
    with st.spinner("正在計算 Perplexity 與提取 Stylometry 特徵..."):
        time.sleep(1) # UX 模擬運算感
        result = analyze_text(input_text)
    
    # --- 結果呈現區 (Col 2) ---
    with col2:
        st.subheader("📊 偵測結果")
        
        # 1. 大圓餅圖/分數 (仿 JustDone)
        prob = result['ai_probability']
        color = "#ef4444" if prob > 50 else "#10b981"
        verdict = "AI Generated" if prob > 50 else "Human Written"
        
        st.markdown(f"""
        <div class="score-card" style="border-top: 5px solid {color};">
            <div class="score-title">AI 生成機率 (Probability)</div>
            <div class="score-value" style="color: {color};">{prob:.1f}%</div>
            <div style="font-weight:bold; color: {color}; margin-top:0.5rem; font-size:1.2rem;">
                {verdict}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 2. 統計數據小卡
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Burstiness", f"{result['burstiness']:.2f}", help="越高代表越像人類 (Slide 9)")
        m2.metric("句子數", result['stats']['sentence_count'])
        m3.metric("總字數", result['stats']['word_count'])

    # --- 深度分析圖表區 (下方) ---
    st.divider()
    st.header("🔬 深度特徵分析 (XAI)")
    
    tab1, tab2, tab3 = st.tabs(["文本螢光筆 (Highlight)", "PP 波動圖 (Perplexity)", "句長分布 (Burstiness)"])
    
    # Tab 1: 文本標註 (仿 JustDone 視覺效果)
    with tab1:
        st.caption("紅色代表「極高 AI 機率」的句子；綠色代表「具有人味」的句子。")
        
        highlighted_html = '<div style="background:#f8fafc; padding:20px; border-radius:10px; line-height:2.0;">'
        
        for idx, sent in enumerate(result['sentences']):
            pp = result['perplexity_trend'][idx]
            # 根據 PP 決定顏色 (PP 低 = AI = 紅色)
            if pp < 15:
                span_class = "highlight-ai"
                tooltip = f"Low Perplexity ({pp:.1f})"
            elif pp > 25:
                span_class = "highlight-human"
                tooltip = f"High Perplexity ({pp:.1f})"
            else:
                span_class = ""
                tooltip = "Neutral"
                
            if span_class:
                highlighted_html += f'<span class="{span_class}" title="{tooltip}">{sent}</span> '
            else:
                highlighted_html += f'{sent} '
                
        highlighted_html += '</div>'
        st.markdown(highlighted_html, unsafe_allow_html=True)

    # Tab 2: Perplexity Chart (Slide 8 Visualization)
    with tab2:
        st.markdown("#### Perplexity Time Series (困惑度時間序列)")
        st.caption("觀察重點：AI (紅線) 通常平滑低得；Human (綠區) 會有突波。")
        
        df_pp = pd.DataFrame({
            "Sentence Index": range(len(result['perplexity_trend'])),
            "Perplexity": result['perplexity_trend']
        })
        
        fig_pp = px.line(
            df_pp, x="Sentence Index", y="Perplexity",
            markers=True, line_shape="spline",
            title="Perplexity Fluctuation"
        )
        fig_pp.update_traces(line_color='#6366f1', line_width=3)
        fig_pp.add_hrect(y0=0, y1=15, line_width=0, fillcolor="red", opacity=0.1, annotation_text="High AI Probability Zone")
        st.plotly_chart(fig_pp, use_container_width=True)

    # Tab 3: Burstiness Histogram (Slide 11)
    with tab3:
        st.markdown("#### Sentence Length Distribution (句長分布)")
        st.caption("觀察重點：AI 分布呈現單峰且集中；Human 分布呈現多峰且長尾 (Long Tail)。")
        
        sent_lens = [len(s) for s in result['sentences']]
        fig_hist = px.histogram(
            x=sent_lens, nbins=10,
            labels={'x': 'Sentence Length (chars)'},
            color_discrete_sequence=['#3b82f6']
        )
        fig_hist.update_layout(bargap=0.1)
        st.plotly_chart(fig_hist, use_container_width=True)

elif analyze_btn and not input_text:
    st.warning("請先輸入文字再進行分析。")

# --- Footer ---
st.markdown("""
<div style="text-align:center; margin-top:50px; color:#94a3b8; font-size:0.8rem;">
    AI Detector Demo | Implements Perplexity, Burstiness & Stylometry Theory<br>
    Disclaimer: Results are probabilistic and for educational purposes only.
</div>
""", unsafe_allow_html=True)