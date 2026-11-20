"""PubMed and OpenTargets retrieval for live causal reasoning."""

from src.utils.console import tr
import os
import re
import requests
from typing import List, Optional, Dict, Any, Tuple
from Bio import Entrez
from dotenv import load_dotenv

load_dotenv()


MODIFIER_REMOVAL_RULES = [

    (r'^Acute\s+', ''),           # Acute bronchitis → bronchitis
    (r'^Chronic\s+', ''),         # Chronic bronchitis → bronchitis
    (r'^Subacute\s+', ''),        # Subacute thyroiditis → thyroiditis

    (r'^Spontaneous\s+', ''),     # Spontaneous rib fracture → rib fracture
    (r'^Traumatic\s+', ''),       # Traumatic pneumothorax → pneumothorax

    (r'^Primary\s+', ''),         # Primary hypertension → hypertension
    (r'^Secondary\s+', ''),       # Secondary hypertension → hypertension
    (r'^Idiopathic\s+', ''),      # Idiopathic pulmonary fibrosis → pulmonary fibrosis

    (r'^Atypical\s+', ''),        # Atypical pneumonia → pneumonia
    (r'^Typical\s+', ''),         # Typical angina → angina

    (r'^Severe\s+', ''),          # Severe asthma → asthma
    (r'^Mild\s+', ''),            # Mild TBI → TBI

    (r'\s*\([^)]*\)\s*$', ''),    # Disease (subtype) → Disease
]


MEDICAL_SYNONYMS = {
    "GERD": "Gastroesophageal reflux disease",
    "Boerhaave": "Boerhaave syndrome",
    "PE": "Pulmonary embolism",
    "MI": "Myocardial infarction",
    "DVT": "Deep vein thrombosis",
    "COPD": "Chronic obstructive pulmonary disease",
    "CHF": "Congestive heart failure",
    "UTI": "Urinary tract infection",
    "TIA": "Transient ischemic attack",
    "SLE": "Systemic lupus erythematosus",
    "ARDS": "Acute respiratory distress syndrome",
    "CAD": "Coronary artery disease",
    "AFib": "Atrial fibrillation",
    "HTN": "Hypertension",
}


def clean_disease_name(disease_name: str) -> str:

    if '/' in disease_name:
        parts = [p.strip() for p in disease_name.split('/')]

        disease_name = max(parts, key=len)

    disease_name = re.sub(r'\s*\([^)]*\)\s*', ' ', disease_name).strip()

    disease_name = re.sub(r'\s+', ' ', disease_name)

    return disease_name

def expand_query_conservative(disease_name: str) -> List[str]:

    cleaned_name = clean_disease_name(disease_name)

    expanded = []

    if cleaned_name != disease_name:
        expanded.append(cleaned_name)
    expanded.append(disease_name)

    for name in [cleaned_name, disease_name]:
        if name.upper() in MEDICAL_SYNONYMS:
            synonym = MEDICAL_SYNONYMS[name.upper()]
            if synonym not in expanded:
                expanded.append(synonym)

        if name in MEDICAL_SYNONYMS:
            synonym = MEDICAL_SYNONYMS[name]
            if synonym not in expanded:
                expanded.append(synonym)

    for name in [cleaned_name, disease_name]:
        for pattern, replacement in MODIFIER_REMOVAL_RULES:
            new_term = re.sub(pattern, replacement, name, flags=re.IGNORECASE).strip()
            if new_term and new_term != name and new_term not in expanded:

                if len(new_term.split()) >= 1 and len(new_term) >= 5:
                    expanded.append(new_term)

    seen = set()
    result = []
    for term in expanded:
        if term.lower() not in seen:
            seen.add(term.lower())
            result.append(term)

    return result

# ==================== Open Targets Platform Tool ====================

class OpenTargetsTool:
    """Retrieve disease information from OpenTargets."""

    API_URL = "https://api.platform.opentargets.org/api/v4/graphql"

    def __init__(self, timeout: int = 30):

        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

        self.last_raw_response: Optional[Dict[str, Any]] = None
        self.last_search_term: str = ""
        print(tr(
            "[OpenTargetsTool] Initialized (with Exact Match validation)",
            '[Open Targets 检索] 已初始化（启用精确匹配验证）',
        ))

    def search_disease_with_validation(
        self,
        disease_name: str,
        search_terms: Optional[List[str]] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:

        from src.utils.knowledge_utils import validate_open_targets_match

        if search_terms is None:
            search_terms = [clean_disease_name(disease_name)]

        self.last_raw_response = None
        self.last_search_term = ""

        for term in search_terms:
            print(tr(
                f"[OpenTargetsTool] Trying search term: '{term}'...",
                f"[Open Targets 检索] 正在尝试检索词：'{term}'...",
            ))
            self.last_search_term = term

            try:

                efo_id, raw_search_response = self._search_disease_id_with_validation(term)
                self.last_raw_response = raw_search_response

                if not efo_id:
                    print(tr(
                        f"[OpenTargetsTool] No valid match for '{term}', trying next synonym...",
                        f"[Open Targets 检索] '{term}' 没有有效匹配，正在尝试下一个同义词...",
                    ))
                    continue

                disease_info = self._get_disease_details(efo_id)
                if disease_info:
                    print(tr(
                        f"[OpenTargetsTool] ✓ Successfully matched: '{term}' -> '{disease_info.get('name', '')}'",
                        f"[Open Targets 检索] ✓ 匹配成功：'{term}' -> '{disease_info.get('name', '')}'",
                    ))
                    return disease_info, raw_search_response

            except Exception as e:
                print(tr(
                    f"[OpenTargetsTool] Error searching '{term}': {e}",
                    f"[Open Targets 检索] 检索 '{term}' 失败：{e}",
                ))
                continue

        print(tr(
            f"[OpenTargetsTool] ✗ No matching disease found after trying all synonyms for '{disease_name}'",
            f"[Open Targets 检索] ✗ 尝试 '{disease_name}' 的所有同义词后，仍未找到匹配疾病",
        ))
        return None, self.last_raw_response

    def search_disease(self, disease_name: str) -> Optional[Dict[str, Any]]:

        result, _ = self.search_disease_with_validation(disease_name)
        return result

    def get_disease_description(self, disease_name: str) -> str:

        result, _ = self.get_disease_description_with_raw(disease_name)
        return result

    def get_disease_description_with_raw(
        self,
        disease_name: str,
        search_terms: Optional[List[str]] = None
    ) -> Tuple[str, Optional[Dict[str, Any]]]:

        disease_info, raw_response = self.search_disease_with_validation(
            disease_name,
            search_terms
        )

        if not disease_info:
            return "", raw_response

        description = disease_info.get("description", "")
        if description:

            name = disease_info.get("name", disease_name)
            result = f"Disease: {name}\n\nDescription: {description}"

            therapeutic_areas = disease_info.get("therapeuticAreas", [])
            if therapeutic_areas:
                areas = [ta.get("name", "") for ta in therapeutic_areas if ta.get("name")]
                if areas:
                    result += f"\n\nTherapeutic Areas: {', '.join(areas)}"

            print(tr(
                f"[OpenTargetsTool] Retrieved description ({len(result)} chars)",
                f'[Open Targets 检索] 已获取描述（{len(result)} 字符）',
            ))
            return result, raw_response

        return "", raw_response

    def _search_disease_id_with_validation(
        self,
        disease_name: str
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:

        from src.utils.knowledge_utils import validate_open_targets_match

        query = """
        query SearchDisease($queryString: String!) {
            search(queryString: $queryString, entityNames: ["disease"], page: {index: 0, size: 5}) {
                hits {
                    id
                    name
                    entity
                }
            }
        }
        """

        variables = {"queryString": disease_name}

        try:
            response = self.session.post(
                self.API_URL,
                json={"query": query, "variables": variables},
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            hits = data.get("data", {}).get("search", {}).get("hits", [])

            for hit in hits:
                if hit.get("entity") != "disease":
                    continue

                returned_name = hit.get("name", "")
                efo_id = hit.get("id", "")

                if validate_open_targets_match(disease_name, returned_name, None):
                    print(tr(
                        f"[OpenTargetsTool] ✓ Validated match: '{disease_name}' -> '{returned_name}' ({efo_id})",
                        f"[Open Targets 检索] ✓ 匹配验证成功：'{disease_name}' -> '{returned_name}'（{efo_id}）",
                    ))
                    return efo_id, data
                else:
                    print(tr(
                        f"[OpenTargetsTool] ✗ Match failed: '{disease_name}' vs '{returned_name}' (skipping)",
                        f"[Open Targets 检索] ✗ 匹配失败：'{disease_name}' 与 '{returned_name}'（已跳过）",
                    ))

            if hits:
                print(tr(
                    f"[OpenTargetsTool] No valid match found in {len(hits)} results",
                    f'[Open Targets 检索] 在 {len(hits)} 条结果中未找到有效匹配',
                ))
            return None, data

        except Exception as e:
            print(tr(
                f"[OpenTargetsTool] Search error: {e}",
                f'[Open Targets 检索] 检索失败：{e}',
            ))
            return None, None

    def _search_disease_id(self, disease_name: str) -> Optional[str]:

        efo_id, _ = self._search_disease_id_with_validation(disease_name)
        return efo_id

    def _get_disease_details(self, efo_id: str) -> Optional[Dict[str, Any]]:

        query = """
        query GetDisease($efoId: String!) {
            disease(efoId: $efoId) {
                id
                name
                description
                synonyms {
                    relation
                    terms
                }
                therapeuticAreas {
                    id
                    name
                }
            }
        }
        """

        variables = {"efoId": efo_id}

        try:
            response = self.session.post(
                self.API_URL,
                json={"query": query, "variables": variables},
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            disease = data.get("data", {}).get("disease")
            return disease

        except Exception as e:
            print(tr(
                f"[OpenTargetsTool] Get details error: {e}",
                f'[Open Targets 检索] 获取详细信息失败：{e}',
            ))
            return None


# ==================== PubMed Tool ====================

class PubMedTool:
    """Retrieve medical publications from PubMed."""

    def __init__(self):
        self.email = os.getenv("PUBMED_EMAIL", "your_email@example.com")
        self.api_key = os.getenv("PUBMED_API_KEY")

        Entrez.email = self.email
        if self.api_key:
            Entrez.api_key = self.api_key
            print(tr(
                f"[PubMedTool] Initialized with API key",
                '[PubMed 检索] 已使用 API 密钥初始化',
            ))
        else:
            print(tr(
                f"[PubMedTool] Initialized without API key (rate limited)",
                '[PubMed 检索] 已在无 API 密钥的模式下初始化（请求速率受限）',
            ))

    def search_general_features(
        self,
        disease_name: str,
        max_results: int = 5,
        use_expansion: bool = True
    ) -> str:

        print(tr(
            f"[PubMedTool] Searching general features for '{disease_name}'...",
            f"[PubMed 检索] 正在检索 '{disease_name}' 的一般特征...",
        ))

        result = self._search_review_guideline(disease_name, max_results)
        if result:
            return result

        result = self._search_general_fallback(disease_name, max_results)
        if result:
            return result

        if use_expansion:
            expanded_terms = expand_query_conservative(disease_name)
            print(tr(
                f"[PubMedTool] Trying Query Expansion: {expanded_terms}",
                f'[PubMed 检索] 正在尝试扩展检索词：{expanded_terms}',
            ))

            for term in expanded_terms[1:]:
                print(tr(
                    f"[PubMedTool] Trying expanded term: '{term}'",
                    f"[PubMed 检索] 正在尝试扩展词：'{term}'",
                ))

                result = self._search_review_guideline(term, max_results)
                if result:
                    print(tr(
                        f"[PubMedTool] Found results with expanded term: '{term}'",
                        f"[PubMed 检索] 扩展词 '{term}' 已找到结果",
                    ))
                    return result

                result = self._search_general_fallback(term, max_results)
                if result:
                    print(tr(
                        f"[PubMedTool] Found fallback results with: '{term}'",
                        f"[PubMed 检索] 备用检索词 '{term}' 已找到结果",
                    ))
                    return result

        print(tr(
            f"[PubMedTool] No results found for '{disease_name}' after all attempts",
            f"[PubMed 检索] 尝试所有检索方式后，未找到 '{disease_name}' 的结果",
        ))
        return ""

    def _search_review_guideline(self, disease_name: str, max_results: int = 5) -> str:

        try:
            query = (
                f'("{disease_name}"[Title]) AND '
                f'("clinical features"[Title/Abstract] OR "signs and symptoms"[Title/Abstract] OR "diagnosis"[Title/Abstract]) AND '
                f'("Review"[Publication Type] OR "Practice Guideline"[Publication Type])'
            )

            search_handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results + 5,
                retmode="xml"
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()

            id_list = search_results.get("IdList", [])
            print(tr(
                f"[PubMedTool] Found {len(id_list)} Review/Guideline articles",
                f'[PubMed 检索] 找到 {len(id_list)} 篇综述或指南文献',
            ))

            if not id_list:
                return ""

            abstracts = self._fetch_abstracts(id_list, max_results)
            if not abstracts:
                return ""

            merged_text = "\n\n---\n\n".join(abstracts)
            print(tr(
                f"[PubMedTool] Merged {len(abstracts)} abstracts ({len(merged_text)} chars)",
                f'[PubMed 检索] 已合并 {len(abstracts)} 篇摘要（{len(merged_text)} 字符）',
            ))
            return merged_text

        except Exception as e:
            print(tr(
                f"[PubMedTool] Review/Guideline search error: {e}",
                f'[PubMed 检索] 综述或指南检索失败：{e}',
            ))
            return ""

    def _search_general_fallback(self, disease_name: str, max_results: int = 5) -> str:

        try:
            query = (
                f'("{disease_name}"[Title/Abstract]) AND '
                f'("clinical features"[All Fields] OR "signs and symptoms"[All Fields] OR "clinical presentation"[All Fields])'
            )

            search_handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results + 5,
                retmode="xml"
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()

            id_list = search_results.get("IdList", [])
            print(tr(
                f"[PubMedTool] Fallback found {len(id_list)} articles",
                f'[PubMed 检索] 备用检索找到 {len(id_list)} 篇文献',
            ))

            if not id_list:
                return ""

            abstracts = self._fetch_abstracts(id_list, max_results)
            if not abstracts:
                return ""

            merged_text = "\n\n---\n\n".join(abstracts)
            print(tr(
                f"[PubMedTool] Fallback merged {len(abstracts)} abstracts ({len(merged_text)} chars)",
                f'[PubMed 检索] 备用检索已合并 {len(abstracts)} 篇摘要（{len(merged_text)} 字符）',
            ))
            return merged_text

        except Exception as e:
            print(tr(
                f"[PubMedTool] Fallback search error: {e}",
                f'[PubMed 检索] 备用检索失败：{e}',
            ))
            return ""

    def _fetch_abstracts(self, id_list: List[str], max_results: int) -> List[str]:

        abstracts = []
        for pmid in id_list[:max_results + 3]:
            abstract_text = self._fetch_abstract(pmid)
            if abstract_text and len(abstract_text) >= 100:
                abstracts.append(abstract_text)
                if len(abstracts) >= max_results:
                    break
        return abstracts

    def search_differential(
        self,
        disease_a: str,
        disease_b: str,
        max_results: int = 3
    ) -> List[str]:

        print(tr(
            f"[PubMedTool] Searching differential: '{disease_a}' vs '{disease_b}'",
            f"[PubMed 检索] 正在检索鉴别诊断：'{disease_a}' 与 '{disease_b}'",
        ))

        try:
            query = (
                f'(("Differential diagnosis"[MeSH Terms] OR "Differential diagnosis"[All Fields]) '
                f'AND ("{disease_a}"[All Fields] AND "{disease_b}"[All Fields]))'
            )

            search_handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results + 5,
                retmode="xml"
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()

            id_list = search_results.get("IdList", [])
            print(tr(
                f"[PubMedTool] Found {len(id_list)} articles",
                f'[PubMed 检索] 找到 {len(id_list)} 篇文献',
            ))

            if not id_list:
                return self._fallback_differential_search(disease_a, disease_b, max_results)

            abstracts = self._fetch_abstracts(id_list, max_results)
            print(tr(
                f"[PubMedTool] Retrieved {len(abstracts)} valid abstracts",
                f'[PubMed 检索] 已获取 {len(abstracts)} 篇有效摘要',
            ))
            return abstracts

        except Exception as e:
            print(tr(
                f"[PubMedTool] Error searching differential: {e}",
                f'[PubMed 检索] 鉴别诊断检索失败：{e}',
            ))
            return []

    def _fallback_differential_search(
        self,
        disease_a: str,
        disease_b: str,
        max_results: int = 3
    ) -> List[str]:

        print(tr(
            f"[PubMedTool] Trying fallback search...",
            '[PubMed 检索] 正在尝试备用检索...',
        ))

        try:
            query = f'("{disease_a}"[All Fields] AND "{disease_b}"[All Fields])'

            search_handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results + 5,
                retmode="xml"
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()

            id_list = search_results.get("IdList", [])
            print(tr(
                f"[PubMedTool] Fallback found {len(id_list)} articles",
                f'[PubMed 检索] 备用检索找到 {len(id_list)} 篇文献',
            ))

            if not id_list:
                return []

            return self._fetch_abstracts(id_list, max_results)

        except Exception as e:
            print(tr(
                f"[PubMedTool] Fallback search error: {e}",
                f'[PubMed 检索] 备用检索失败：{e}',
            ))
            return []

    def _fetch_abstract(self, pmid: str) -> str:

        try:
            fetch_handle = Entrez.efetch(
                db="pubmed",
                id=pmid,
                rettype="abstract",
                retmode="xml"
            )
            fetch_results = Entrez.read(fetch_handle)
            fetch_handle.close()

            article_results = fetch_results.get("PubmedArticle", [])
            if not article_results:
                return ""

            article = article_results[0]

            title = (
                article.get("MedlineCitation", {})
                .get("Article", {})
                .get("ArticleTitle", "")
            )

            abstract_sections = (
                article.get("MedlineCitation", {})
                .get("Article", {})
                .get("Abstract", {})
                .get("AbstractText", [])
            )

            if not abstract_sections:
                return ""

            abstract_parts = []
            for section in abstract_sections:
                if hasattr(section, 'attributes') and section.attributes:
                    label = section.attributes.get("Label", "")
                    if label:
                        abstract_parts.append(f"{label}: {section}")
                    else:
                        abstract_parts.append(str(section))
                else:
                    abstract_parts.append(str(section))

            abstract_text = "\n\n".join(abstract_parts)

            if title:
                return f"Title: {title}\n\n{abstract_text}"

            return abstract_text

        except Exception as e:
            print(tr(
                f"[PubMedTool] Error fetching abstract for PMID {pmid}: {e}",
                f'[PubMed 检索] 获取 PMID {pmid} 的摘要失败：{e}',
            ))
            return ""
