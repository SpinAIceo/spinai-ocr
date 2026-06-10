"""Receipt structured parsing from OCR text lines.

Extracts store name, date, items, total, payment info from Korean receipt
OCR output using regex pattern matching.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ReceiptItem:
    name: str
    unit_price: int | None = None
    qty: int | None = None
    amount: int | None = None


@dataclass
class ReceiptPayment:
    method: str = ""
    card: str = ""
    card_number: str = ""
    amount: int | None = None
    installment: str = ""


@dataclass
class Receipt:
    store_name: str = ""
    business_number: str = ""
    phone: str = ""
    address: str = ""
    date: str = ""
    time: str = ""
    receipt_number: str = ""
    items: list[ReceiptItem] = field(default_factory=list)
    total: int | None = None
    payment: ReceiptPayment = field(default_factory=ReceiptPayment)
    raw_lines: list[str] = field(default_factory=list)


_DATE_RE = re.compile(r'(\d{4})[년./-](\d{1,2})[월./-](\d{1,2})')
_TIME_RE = re.compile(r'(\d{1,2}):(\d{2})(?::(\d{2}))?')
_PHONE_RE = re.compile(r'(\d{2,4}-\d{3,4}-\d{4})')
_PRICE_RE = re.compile(r'([\d,]+)\s*원?$')
_RECEIPT_NO_RE = re.compile(r'No[:.]\s*(\d+)', re.IGNORECASE)
_BIZ_NO_RE = re.compile(r'(\d{3}-?\d{2}-?\d{5})')
_CARD_NAMES = ['비씨', '삼성', '신한', '현대', '롯데', '국민', '농협', '하나', 'IBK', '마스타', '마스터', '비자', 'VISA']
_CARD_RE = re.compile(r'(' + '|'.join(_CARD_NAMES) + r').*(?:카드)?', re.IGNORECASE)
_STORE_INDICATORS = ['상호', '상 호', '매장', '가맹점', '가 맹 점']
_CARD_NUM_RE = re.compile(r'(\d{4})[- ](\d{4})[- ]\*{2,4}[- ]\*{2,4}')
_ITEM_FULL_RE = re.compile(r'^(.+?)\s+([\d,]+)\s+(\d+)\s+([\d,]+)\s*$')
_ITEM_SIMPLE_RE = re.compile(r'^(.+?)\s+([\d,]+)\s*$')


def _parse_price(s: str) -> int | None:
    m = _PRICE_RE.search(s)
    if m:
        try:
            return int(m.group(1).replace(',', ''))
        except ValueError:
            pass
    nums = re.findall(r'[\d,]+', s)
    for n in reversed(nums):
        try:
            v = int(n.replace(',', ''))
            if v > 0:
                return v
        except ValueError:
            pass
    return None


def _try_parse_item(text: str) -> ReceiptItem | None:
    s = text.strip()
    skip_kw = ['판매', '합계', '부가', '거스름', '카드', '승인', '전표', '할부',
               '영수', '상 품', '단가', '수량', '금액', '받올', '받물', '과세']
    m = _ITEM_FULL_RE.match(s)
    if m:
        name = m.group(1).strip()
        if any(kw in name for kw in skip_kw):
            return None
        try:
            return ReceiptItem(
                name=name,
                unit_price=int(m.group(2).replace(',', '')),
                qty=int(m.group(3)),
                amount=int(m.group(4).replace(',', '')),
            )
        except ValueError:
            return None
    m = _ITEM_SIMPLE_RE.match(s)
    if m:
        name = m.group(1).strip()
        if any(kw in name for kw in skip_kw) or len(name) < 2:
            return None
        try:
            amount = int(m.group(2).replace(',', ''))
        except ValueError:
            return None
        if amount <= 0:
            return None
        return ReceiptItem(name=name, amount=amount)
    return None


def parse_receipt(lines: list[str]) -> Receipt:
    """Parse OCR lines into a structured Receipt."""
    r = Receipt(raw_lines=lines)

    for line in lines:
        text = line.strip()
        if not text:
            continue

        if not r.date:
            m = _DATE_RE.search(text)
            if m:
                r.date = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

        if not r.time:
            m = _TIME_RE.search(text)
            if m:
                r.time = f"{m.group(1)}:{m.group(2)}"

        if not r.phone:
            m = _PHONE_RE.search(text)
            if m:
                r.phone = m.group(1)

        if not r.receipt_number:
            m = _RECEIPT_NO_RE.search(text)
            if m:
                r.receipt_number = m.group(1)

        if not r.business_number:
            m = _BIZ_NO_RE.search(text)
            if m:
                r.business_number = m.group(1)

        if '승인' in text and '카드' in text:
            pass
        else:
            m = _CARD_RE.search(text)
            if m and not r.payment.card:
                r.payment.card = m.group(0).strip()
                r.payment.method = "신용카드"

        if '판매금액' in text or '합계' in text:
            r.total = _parse_price(text)

        if '일시불' in text:
            r.payment.installment = "일시불"

        if '신용카드' in text and r.payment.amount is None:
            r.payment.amount = _parse_price(text)

        if not r.payment.card_number:
            m = _CARD_NUM_RE.search(text)
            if m:
                r.payment.card_number = m.group(0)

        for ind in _STORE_INDICATORS:
            if ind in text and not r.store_name:
                parts = text.split(ind, 1)
                if len(parts) > 1:
                    r.store_name = parts[1].strip().strip(':').strip()

        item = _try_parse_item(text)
        if item:
            r.items.append(item)

    if not r.total and r.payment.amount:
        r.total = r.payment.amount

    return r
