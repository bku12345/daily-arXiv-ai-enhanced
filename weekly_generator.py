import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import os
from openai import OpenAI

# 初始化 OpenAI 客户端（读取 GitHub Actions 的 Secrets）
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

# 配置项（可通过 GitHub Variables 调整）
BASE_URL = "https://bku12345.github.io/daily-arXiv-ai-enhanced/"
WEEK_DAYS = 7
# 读取 GitHub Variables 中的分类（和原项目保持一致）
TARGET_CATEGORIES = os.getenv("CATEGORIES", "cs.CV,cs.GR,cs.CL,cs.AI").split(",")
LANGUAGE = os.getenv("LANGUAGE", "Chinese")

def get_daily_papers(date_str: str) -> list:
    """爬取指定日期的论文（适配原项目真实HTML结构）"""
    try:
        daily_url = f"{BASE_URL}{date_str}.html"
        response = requests.get(daily_url, timeout=15)
        if response.status_code != 200:
            return []
        
        soup = BeautifulSoup(response.text, "html.parser")
        papers = []
        
        # 适配原项目真实的论文卡片选择器（核心调整）
        # 先找到所有论文卡片的容器（根据原项目页面结构）
        paper_items = soup.find_all("div", class_="card mb-3")  # 原项目真实的论文卡片类名
        for item in paper_items:
            # 解析标题和链接（适配真实标签结构）
            title_elem = item.find("h4") or item.find("h5")
            if not title_elem:
                continue
            title = title_elem.text.strip()
            url_elem = title_elem.find("a")
            url = url_elem["href"] if url_elem and "href" in url_elem.attrs else ""
            
            # 解析摘要（适配真实标签）
            abstract_elem = item.find("div", class_="card-body") or item.find("p")
            abstract = abstract_elem.text.strip() if abstract_elem else ""
            
            # 解析分类/作者（适配真实标签）
            meta_elem = item.find("small", class_="text-muted")
            meta_text = meta_elem.text.strip() if meta_elem else ""
            
            # 过滤目标分类的论文
            if any(cat.strip() in meta_text for cat in TARGET_CATEGORIES):
                papers.append({
                    "date": date_str,
                    "title": title,
                    "abstract": abstract,
                    "meta": meta_text,
                    "url": url
                })
        return papers
    except Exception as e:
        print(f"爬取 {date_str} 失败: {str(e)}")
        return []
        
def get_weekly_papers() -> tuple:
    """获取过去7天的论文"""
    end_dt = datetime.now()
    date_list = [(end_dt - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(WEEK_DAYS)]
    weekly_papers = []
    
    for date_str in date_list:
        daily_papers = get_daily_papers(date_str)
        weekly_papers.extend(daily_papers)
    
    # 按分类聚合
    categorized_papers = {}
    for paper in weekly_papers:
        # 提取分类
        cat = [c for c in TARGET_CATEGORIES if c.strip() in paper["meta"]]
        cat = cat[0] if cat else "Other"
        if cat not in categorized_papers:
            categorized_papers[cat] = []
        categorized_papers[cat].append(paper)
    
    return weekly_papers, categorized_papers

def generate_weekly_report(categorized_papers: dict) -> str:
    """生成周报（适配语言配置）"""
    prompt = f"""
    请用{LANGUAGE}生成arXiv每周AI论文汇总周报，要求：
    1. 整体总结：本周{list(categorized_papers.keys())}分类论文的核心趋势、热门研究方向（150字左右）
    2. 分类详情：按分类总结关键研究内容、创新点（每个分类100字左右）
    3. 值得关注的论文：列出3-5篇有重要突破的论文（标题+核心贡献）
    4. 语言简洁专业，结构清晰，适合科研人员快速阅读

    论文数据：{categorized_papers}
    """
    
    try:
        response = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "deepseek-chat"),
            messages=[
                {"role": "system", "content": "你是专业的AI领域研究员，擅长总结arXiv论文并生成结构化周报"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"周报生成失败：{str(e)}\n请检查API Key和BASE_URL配置"

def save_files(weekly_papers: list, report: str):
    """保存论文数据和周报（修复参数错误）"""
    # 保存论文数据为JSON（移除ensure_ascii，改用force_ascii=False）
    df = pd.DataFrame(weekly_papers)
    df.to_json("weekly_papers.json", orient="records", force_ascii=False, indent=2)
    
    # 保存周报为MD
    with open("weekly_report.md", "w", encoding="utf-8") as f:
        f.write(f"# arXiv 每周论文汇总 ({datetime.now().strftime('%Y-%m-%d')})\n\n{report}")
if __name__ == "__main__":
    # 主流程
    print("开始爬取每周论文...")
    weekly_papers, categorized_papers = get_weekly_papers()
    print(f"爬取到 {len(weekly_papers)} 篇论文")
    
    print("生成周报...")
    report = generate_weekly_report(categorized_papers)
    
    print("保存文件...")
    save_files(weekly_papers, report)
    print("周报生成完成！")
