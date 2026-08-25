"""Merchants and narration templates.

The bank statement is the messy source, and its mess has to be *generated*
rather than hand-waved. Every merchant carries several name variants -- the
abbreviated, vowel-dropped forms an Indian bank statement actually contains --
because resolving them is what T3 exists to do.

These variants are also the argument for cutting ChromaDB. ``RZRPAY SFTWR P L``
does not sit near ``Razorpay Software Private Limited`` in sentence-embedding
space; MiniLM has no reason to relate a consonant skeleton to its expansion.
Stripping legal suffixes and fuzzy-matching the skeleton does relate them.
"""

from __future__ import annotations

from typing import Final

from ledgerloop.models.base import FrozenLedgerModel

__all__ = ["MERCHANTS", "NARRATION_WITHOUT_UTR", "NARRATION_WITH_UTR", "NOISE_NARRATIONS"]


class Merchant(FrozenLedgerModel):
    """One merchant, with the names a bank statement might use for it."""

    merchant_id: str
    legal_name: str
    variants: tuple[str, ...]


MERCHANTS: Final[tuple[Merchant, ...]] = (
    Merchant(
        merchant_id="MRCH_0001",
        legal_name="Razorpay Software Private Limited",
        variants=("RAZORPAY SOFTWARE PVT", "RZRPAY SFTWR P L", "RAZORPAY SOFTWARE PRIVATE LTD"),
    ),
    Merchant(
        merchant_id="MRCH_0002",
        legal_name="Nykaa E-Retail Private Limited",
        variants=("NYKAA E RETAIL PVT LTD", "NYKAA ERETAIL P L", "NYKAA E-RETAIL PRIVATE"),
    ),
    Merchant(
        merchant_id="MRCH_0003",
        legal_name="Zomato Hyperpure Private Limited",
        variants=("ZOMATO HYPERPURE PVT", "ZMTO HYPRPURE P L", "ZOMATO HYPERPURE PRIVATE LTD"),
    ),
    Merchant(
        merchant_id="MRCH_0004",
        legal_name="Urban Company Technologies Limited",
        variants=("URBAN COMPANY TECH LTD", "URBN CO TECHNOLOGIES", "URBANCOMPANY TECH LIMITED"),
    ),
    Merchant(
        merchant_id="MRCH_0005",
        legal_name="Lenskart Solutions Private Limited",
        variants=("LENSKART SOLUTIONS PVT", "LNSKRT SOLTNS P L", "LENSKART SOLUTIONS PRIVATE"),
    ),
    Merchant(
        merchant_id="MRCH_0006",
        legal_name="Boat Lifestyle Retail Private Limited",
        variants=("BOAT LIFESTYLE RETAIL", "BOAT LFSTYL RTL P L", "BOAT LIFESTYLE PVT LTD"),
    ),
    Merchant(
        merchant_id="MRCH_0007",
        legal_name="Cred Avenues Technologies Limited",
        variants=("CRED AVENUES TECH LTD", "CRED AVNUES TCH", "CREDAVENUES TECHNOLOGIES"),
    ),
    Merchant(
        merchant_id="MRCH_0008",
        legal_name="Meesho Commerce Private Limited",
        variants=("MEESHO COMMERCE PVT", "MSHO COMMRC P L", "MEESHO COMMERCE PRIVATE LTD"),
    ),
    Merchant(
        merchant_id="MRCH_0009",
        legal_name="Swiggy Instamart Services Limited",
        variants=("SWIGGY INSTAMART SVCS", "SWGY INSTMRT SRVCS", "SWIGGY INSTAMART LTD"),
    ),
    Merchant(
        merchant_id="MRCH_0010",
        legal_name="Dream Sports Ventures Private Limited",
        variants=("DREAM SPORTS VENTURES", "DRM SPRTS VNTRS P L", "DREAMSPORTS VENTURES PVT"),
    ),
    Merchant(
        merchant_id="MRCH_0011",
        legal_name="Groww Invest Tech Private Limited",
        variants=("GROWW INVEST TECH PVT", "GRWW INVST TCH P L", "GROWW INVESTTECH PRIVATE"),
    ),
    Merchant(
        merchant_id="MRCH_0012",
        legal_name="Zerodha Broking Limited",
        variants=("ZERODHA BROKING LTD", "ZRDHA BRKNG L", "ZERODHA BROKING LIMITED"),
    ),
)

#: Narration shapes carrying a recoverable UTR. T0 resolves these.
NARRATION_WITH_UTR: Final[tuple[str, ...]] = (
    "NEFT CR-{variant}-{utr}-SETTLEMENT",
    "IMPS CR/{utr}/{variant}/PAYOUT",
    "RTGS CR {variant} {utr} SETTLEMENT",
    "NEFT-{utr}-{variant}-MERCHANT PAYOUT",
    "CR/NEFT/{utr}/{variant}",
)

#: Narration shapes with the reference stripped out entirely -- anomaly A07.
#: Only the merchant name survives, so resolution has to go through T3.
NARRATION_WITHOUT_UTR: Final[tuple[str, ...]] = (
    "NEFT CR-{variant}-SETTLEMENT",
    "IMPS CR/{variant}/PAYOUT",
    "RTGS CR {variant} MERCHANT SETTLEMENT",
    "CR/NEFT/{variant}/BULK",
)

#: Rows that must match nothing. A system that links one of these has produced a
#: false positive, and the evaluator counts it as such.
NOISE_NARRATIONS: Final[tuple[str, ...]] = (
    "RENT PAYMENT COMMERCIAL PREMISES",
    "SALARY CREDIT PAYROLL BATCH",
    "GST PAYMENT CHALLAN",
    "ELECTRICITY BILL BESCOM",
    "VENDOR PAYMENT OFFICE SUPPLIES",
    "INSURANCE PREMIUM CORPORATE",
    "INTERNET LEASED LINE CHARGES",
    "TDS REMITTANCE Q4",
    "CASH DEPOSIT BRANCH COUNTER",
    "INTEREST CREDIT SAVINGS",
)
