from .protocol import Paper
import math
from html import escape


framework = """
<!DOCTYPE HTML>
<html>
<head>
  <style>
    .star-wrapper {
      font-size: 1.3em; /* 调整星星大小 */
      line-height: 1; /* 确保垂直对齐 */
      display: inline-flex;
      align-items: center; /* 保持对齐 */
    }
    .half-star {
      display: inline-block;
      width: 0.5em; /* 半颗星的宽度 */
      overflow: hidden;
      white-space: nowrap;
      vertical-align: middle;
    }
    .full-star {
      vertical-align: middle;
    }
  </style>
</head>
<body>

<div>
    __CONTENT__
</div>

<br><br>
<div>
To unsubscribe, remove your email in your Github Action setting.
</div>

</body>
</html>
"""

def get_empty_html():
  block_template = """
  <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
 <tr>
   <td style="font-size: 20px; font-weight: bold; color: #333;">
        今天没有符合条件的论文，休息一下！ / No Papers Today. Take a rest!
   </td>
 </tr>
  </table>
  """
  return block_template

def get_block_html(
    title: str,
    authors: str,
    rate: str,
    tldr: str,
    pdf_url: str,
    affiliations: str = None,
    *,
    bucket: str = "",
    categories: str = "",
    contribution_en: str = "",
    why_care_zh: str = "",
    why_care_en: str = "",
    paper_url: str = "",
):
    title = escape(title or "")
    authors = escape(authors or "")
    rate = escape(str(rate))
    tldr = escape(tldr or "")
    pdf_url = escape(pdf_url or paper_url or "#", quote=True)
    paper_url = escape(paper_url or "#", quote=True)
    affiliations = escape(affiliations or "Unknown Affiliation")
    bucket = escape(bucket.title())
    categories = escape(categories)
    contribution_en = escape(contribution_en or "")
    why_care_zh = escape(why_care_zh or "")
    why_care_en = escape(why_care_en or "")
    block_template = """
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
    <tr>
        <td style="font-size: 20px; font-weight: bold; color: #333;">
            {title}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #666; padding: 8px 0;">
            {authors}
            <br>
            <i>{affiliations}</i>
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>分栏 / Category:</strong> {bucket}
            <br><strong>分类 / arXiv:</strong> {categories}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>相关性 / Relevance:</strong> {rate}/10
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>贡献 / Contribution:</strong> {tldr}
            <br><i>{contribution_en}</i>
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>为什么值得看 / Why you might care:</strong> {why_care_zh}
            <br><i>{why_care_en}</i>
        </td>
    </tr>

    <tr>
        <td style="padding: 8px 0;">
            <a href="{paper_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #666; padding: 8px 16px; border-radius: 4px;">arXiv</a>
            <a href="{pdf_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #d9534f; padding: 8px 16px; border-radius: 4px;">PDF</a>
        </td>
    </tr>
</table>
"""
    return block_template.format(
        title=title,
        authors=authors,
        rate=rate,
        tldr=tldr,
        pdf_url=pdf_url,
        affiliations=affiliations,
        bucket=bucket,
        categories=categories,
        contribution_en=contribution_en,
        why_care_zh=why_care_zh,
        why_care_en=why_care_en,
        paper_url=paper_url,
    )

def get_stars(score:float):
    full_star = '<span class="full-star">⭐</span>'
    half_star = '<span class="half-star">⭐</span>'
    low = 6
    high = 8
    if score <= low:
        return ''
    elif score >= high:
        return full_star * 5
    else:
        interval = (high-low) / 10
        star_num = math.ceil((score-low) / interval)
        full_star_num = int(star_num/2)
        half_star_num = star_num - full_star_num * 2
        return '<div class="star-wrapper">'+full_star * full_star_num + half_star * half_star_num + '</div>'


def render_email(papers:list[Paper]) -> str:
    parts = []
    if len(papers) == 0 :
        return framework.replace('__CONTENT__', get_empty_html())
    
    for p in papers:
        #rate = get_stars(p.score)
        rate = round(p.score, 1) if p.score is not None else 'Unknown'
        author_list = [a for a in p.authors]
        num_authors = len(author_list)
        if num_authors <= 5:
            authors = ', '.join(author_list)
        else:
            authors = ', '.join(author_list[:3] + ['...'] + author_list[-2:])
        if p.affiliations is not None:
            affiliations = p.affiliations[:5]
            affiliations = ', '.join(affiliations)
            if len(p.affiliations) > 5:
                affiliations += ', ...'
        else:
            affiliations = 'Unknown Affiliation'
        assessment = p.assessment or {}
        contribution = assessment.get("contribution", {})
        why_care = assessment.get("why_care", {})
        parts.append(
            get_block_html(
                p.title,
                authors,
                rate,
                p.tldr,
                p.pdf_url,
                affiliations,
                bucket=assessment.get("bucket", ""),
                categories=", ".join(p.categories or []),
                contribution_en=contribution.get("en", ""),
                why_care_zh=why_care.get("zh", ""),
                why_care_en=why_care.get("en", ""),
                paper_url=p.url,
            )
        )

    content = '<br>' + '</br><br>'.join(parts) + '</br>'
    return framework.replace('__CONTENT__', content)
