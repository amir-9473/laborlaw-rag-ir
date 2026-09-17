"""Local history resolution: no extra model call for follow-ups or recall."""
import re
from collections.abc import Mapping


def recent_questions(history, max_turns, max_chars):
    messages = [m for m in (history or []) if isinstance(m, Mapping) and m.get('role') in {'user', 'assistant'} and isinstance(m.get('content'), str)]
    selected=[];remaining=max_chars
    for message in reversed(messages[-max_turns*2:]):
        text=' '.join(message['content'].split())
        if not text or remaining<=0:continue
        text=text[:remaining];remaining-=len(text)
        if message['role']=='user':selected.append(text)
    return list(reversed(selected))


def recall_previous(query, questions):
    question = r'(?:سوال|سؤال|پرسش)'
    previous = r'(?:قبلی|پیشین|آخرین|قبلم|قبلیم)'
    recall = re.search(question+r'\s+(?:من\s+)?'+previous, query) or re.search(r'آخرین\s+'+question, query)
    starts_with_recall = re.search(r'^(?:(?:لطفا|لطفاً|بگو|به من بگو)\s+)?(?:'+question+r'|آخرین\s+'+question+r')\b', query)
    asks_what = re.search(r'(?:چی|چه)(?:\s+چیزی)?\s+بود[؟?!.\s]*$|(?:چیست|چیه)[؟?!.\s]*$|(?:یادآوری|تکرار)\s*(?:کن|کنید)[؟?!.\s]*$', query)
    if len(query.split()) <= 16 and starts_with_recall and recall and asks_what:
        return f'پرسش قبلی شما این بود:\n\n«{questions[-1]}»' if questions else 'هنوز پرسش قبلی در این گفتگو ثبت نشده است.'
    if re.search(r'قبل از این.*(?:پرسیدم|سوال کردم|سؤال کردم)',query):
        return f'پرسش قبلی شما این بود:\n\n«{questions[-1]}»' if questions else 'هنوز پرسش قبلی در این گفتگو ثبت نشده است.'
    return None


def resolve_followup(query, questions, is_labor):
    refers = re.search(r'^(?:و|پس|همچنین|حالا|در ادامه|ادامه)\b|\b(?:قبلی|همون|همان|این|آن|اون|او|ایشان)\b',query)
    elliptical = len(query.split()) <= 10 and not is_labor(query)
    if not (refers or elliptical):return query
    anchor=next((q for q in reversed(questions) if is_labor(q)),None)
    if not anchor:return query
    room=max(0,2000-len(query)-2)
    anchor=anchor[:min(800,room)]
    return f'{anchor}\n{query}' if anchor else query


def generation_question(query, history):
    if not history:return query
    return f'''تاریخچهٔ محدود همین گفتگو (فقط برای فهم ارجاع‌ها؛ منبع حکم حقوقی نیست):
{history}

پرسش فعلی کاربر:
{query}

فقط به پرسش فعلی پاسخ دهید. پاسخ حقوقی را صرفاً از منابع بازیابی‌شده مستند کنید، نه از پاسخ‌های قبلی.'''
