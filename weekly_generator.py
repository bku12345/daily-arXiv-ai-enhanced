import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import openai

# 加载环境变量
load_dotenv()

# 配置项（可根据需求调整）
TARGET_CATEGORIES = ["All ,cs.AI ,cs.CE ,cs.CL ,cs.CV ,cs.GT ,cs.IT"]  # 目标论文分类
WEEK_DAYS = 7  # 爬取近7天的论文
LANGUAGE = os.getenv("LANGUAGE", "Chinese or English")  # 周报生成语言

# 初始化 OpenAI/DeepSeek 客户端
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.base_url = os.getenv("OPENAI_BASE_URL", "https://api.silicon.com")
MODEL_NAME = os.getenv("MODEL_NAME", "Weekly arXiv AI Enhanced")

def get_daily_papers(date_str: str) -> list:
    """
    爬取指定日期的每日论文页面数据（适配原项目真实HTML结构）
    :param date_str: 日期字符串，格式 YYYY-MM-DD
    :return: 论文列表
    """
    try:
        # 原项目每日论文页面的URL格式
        daily_url = f"https://bku12345.github.io/daily-arXiv-ai-enhanced/{date_str}.html"
        response = requests.get(daily_url, timeout=15)
        
        # 页面无法访问则返回空列表
        if response.status_code != 200:
            print(f"⚠️  {date_str} 页面无法访问，状态码：{response.status_code}")
            return []
        
        soup = BeautifulSoup(response.text, "html.parser")
        papers = []
        
        # 原项目真实的论文卡片类名：col-md-6 col-lg-4 mb-4
        paper_items = soup.find_all("div", class_="col-md-6 col-lg-4 mb-4")
        for item in paper_items:
            # 解析标题和链接（原项目标题在h5.card-title）
            title_elem = item.find("h5", class_="card-title")
            title = title_elem.text.strip() if title_elem else ""
            url = title_elem.find("a")["href"] if (title_elem and title_elem.find("a")) else ""
            
            # 解析摘要（原项目摘要在div.card-text）
            abstract_elem = item.find("div", class_="card-text")
            abstract = abstract_elem.text.strip() if abstract_elem else ""
            
            # 解析作者/分类（原项目在small标签）
            meta_elem = item.find("small")
            meta_text = meta_elem.text.strip() if meta_elem else ""
            
            # 过滤目标分类的论文
            if any(cat.strip() in meta_text for cat in TARGET_CATEGORIES):
                papers.append({
                    "date": date_str,
                    "title": title,
                    "abstract": abstract,
                    "meta": meta_text,  # 作者+分类信息
                    "url": url
                })
        
        print(f"✅ {date_str} 爬取到 {len(papers)} 篇目标论文")
        return papers
    except Exception as e:
        print(f"❌ 爬取{date_str}失败：{str(e)}")
        return []

def get_weekly_papers() -> tuple[list, dict]:
    """
    爬取近7天的所有目标论文，并按分类整理
    :return: 所有论文列表、按分类分组的论文字典
    """
    weekly_papers = []
    categorized_papers = {cat: [] for cat in TARGET_CATEGORIES}
    
    # 生成近7天的日期字符串（YYYY-MM-DD）
    for i in range(WEEK_DAYS):
        target_date = datetime.now() - timedelta(days=i)
        date_str = target_date.strftime("%Y-%m-%d")
        daily_papers = get_daily_papers(date_str)
        weekly_papers.extend(daily_papers)
        
        # 按分类分组
        for paper in daily_papers:
            for cat in TARGET_CATEGORIES:
                if cat in paper["meta"]:
                    categorized_papers[cat].append(paper)
                    break
    
    return weekly_papers, categorized_papers

def generate_weekly_report(categorized_papers: dict) -> str:
    """
    调用大模型生成周报（适配DeepSeek/OpenAI）
    :param categorized_papers: 按分类分组的论文字典
    :return: 生成的周报文本
    """
    # 构造提示词
    prompt = f"""
    请你作为AI领域研究员，用{LANGUAGE}生成arXiv每周论文周报，要求如下：
    1. 整体总结：本周AI/机器学习领域的核心研究趋势（150字左右）；
    2. 分类详情：按{list(categorized_papers.keys())}分别总结，每类突出3-5个核心创新点；
    3. 值得关注的论文：从所有论文中选3-5篇，列出标题+核心贡献（50字/篇）；
    4. 语言简洁专业，符合学术周报风格，不要冗余内容。
    
    论文数据：
    {categorized_papers}
    """
    
    try:
        # 调用DeepSeek/OpenAI API
        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是专业的AI领域研究员，擅长总结arXiv论文周报"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # 降低随机性，保证总结准确
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ 大模型生成周报失败：{str(e)}")
        # 生成兜底周报
        category_details = "\n".join([f"### {cat}\n- 本周共{len(papers)}篇相关论文" for cat, papers in categorized_papers.items()])
        return f"""# arXiv 每周论文汇总 ({datetime.now().strftime('%Y-%m-%d')})

## 整体总结
本周未成功生成AI领域研究趋势总结（原因：{str(e)}）。

## 分类详情
{category_details}

## 值得关注的论文
暂无（生成失败）
"""

def save_files(weekly_papers: list, report: str):
    """
    保存论文数据到JSON、周报到MD（修复pandas参数错误）
    :param weekly_papers: 所有论文列表
    :param report: 生成的周报文本
    """
    try:
        # 保存论文数据到JSON（修复：ensure_ascii → force_ascii）
        df = pd.DataFrame(weekly_papers)
        df.to_json(
            "weekly_papers.json",
            orient="records",
            force_ascii=False,  # 关键修复：支持中文
            indent=2  # 格式化输出，方便查看
        )
        
        # 保存周报到MD
        with open("weekly_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        
        print(f"✅ 文件保存成功：weekly_papers.json（{len(weekly_papers)}条数据）、weekly_report.md")
    except Exception as e:
        print(f"❌ 文件保存失败：{str(e)}")

if __name__ == "__main__":
    """主执行逻辑"""
    print("===== 开始生成arXiv每周论文周报 =====")
    
    # 1. 爬取每周论文
    weekly_papers, categorized_papers = get_weekly_papers()
    total_papers = len(weekly_papers)
    print(f"\n📊 本周共爬取到 {total_papers} 篇目标论文")
    
    # 2. 生成周报（空数据兜底）
    if total_papers == 0:
        print("⚠️  未爬取到任何论文，生成空周报")
        report = f"""# arXiv 每周论文汇总 ({datetime.now().strftime('%Y-%m-%d')})

## 整体总结
本周未爬取到 cs.AI/cs.LG/stat.ML 分类的相关论文，请检查：
1. 原项目每日论文页面是否正常访问；
2. 目标分类是否正确；
3. 网络是否能访问arXiv相关页面。

## 分类详情
- cs.AI：0篇
- cs.LG：0篇
- stat.ML：0篇

## 值得关注的论文
暂无
"""
    else:
        print("📝 开始生成周报...")
        report = generate_weekly_report(categorized_papers)
    
    # 3. 保存文件
    save_files(weekly_papers, report)
    print("\n===== 周报生成流程结束 =====")
