from __future__ import annotations

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup


class LaborLawParser:
    """
    Parser for the Iranian Labor Law source.

    This parser is based on the original working parser
    used in Notebook 1.

    The purpose of this class is to move the parsing logic
    from the notebook into the reusable src package.

    Output structure:

        Article
            ├── Main Article Text
            └── Subarticles / Notes

    Articles and subarticles are stored as separate records
    but are linked through article_number.

    Notebook 2 will later combine them into a single
    Legal Unit per article.
    """

    def __init__(
        self,
        html: str,
        source_url: str | None = None,
        law_title: str = "قانون کار",
    ):
        self.html = html

        self.source_url = source_url

        self.law_title = law_title

        self.soup = BeautifulSoup(
            html,
            "html.parser",
        )


    # =========================================================
    # Main Parsing Method
    # =========================================================

    def parse(
        self,
    ) -> list[dict]:
        """
        Parse the HTML source.

        Returns
        -------
        list[dict]
            Parsed legal records.
        """

        records = []

        current_chapter = None

        current_section = None

        current_article = None

        current_subarticle = None

        # -----------------------------------------------------
        # Extract text from HTML
        # -----------------------------------------------------

        text_elements = self._extract_text_elements()


        # -----------------------------------------------------
        # Process elements sequentially
        # -----------------------------------------------------

        for element in text_elements:

            text = self._clean_text(
                element.get_text(
                    " ",
                    strip=True,
                )
            )

            if not text:
                continue


            # =================================================
            # Chapter
            # =================================================

            chapter_match = re.match(
                r"^\s*فصل\s+(.+?)\s*$",
                text,
            )

            if chapter_match:

                current_chapter = text

                current_section = None

                current_article = None

                current_subarticle = None

                continue


            # =================================================
            # Section / Mabhath
            # =================================================

            section_match = re.match(
                r"^\s*مبحث\s+(.+?)\s*$",
                text,
            )

            if section_match:

                current_section = text

                current_article = None

                current_subarticle = None

                continue


            # =================================================
            # Article
            # =================================================

            article_match = re.match(
                r"^\s*ماده\s+(\d+)\s*[:：\-]?\s*(.*)",
                text,
            )

            if article_match:

                article_number = int(
                    article_match.group(1)
                )

                article_text = (
                    article_match.group(2)
                    .strip()
                )


                current_article = {
                    "record_id": (
                        f"article-{article_number}"
                    ),

                    "law_title": (
                        self.law_title
                    ),

                    "type": "article",

                    "article_number": (
                        article_number
                    ),

                    "article_reference": (
                        f"ماده {article_number}"
                    ),

                    "chapter_title": (
                        current_chapter
                    ),

                    "section_title": (
                        current_section
                    ),

                    "text": (
                        article_text
                    ),

                    "source_url": (
                        self.source_url
                    ),
                }


                records.append(
                    current_article
                )


                current_subarticle = None

                continue


            # =================================================
            # Subarticle / Note
            # =================================================

            subarticle_match = re.match(
                r"^\s*تبصره\s*(\d+)?\s*[:：\-]?\s*(.*)",
                text,
            )


            if (
                subarticle_match
                and current_article is not None
            ):

                subarticle_number = (
                    subarticle_match.group(1)
                )

                subarticle_text = (
                    subarticle_match.group(2)
                    .strip()
                )


                if subarticle_number:

                    subarticle_number = int(
                        subarticle_number
                    )

                else:

                    subarticle_number = (
                        self._get_next_subarticle_number(
                            records=records,
                            article_number=(
                                current_article[
                                    "article_number"
                                ]
                            ),
                        )
                    )


                current_subarticle = {

                    "record_id": (
                        self._build_subarticle_id(
                            article_number=(
                                current_article[
                                    "article_number"
                                ]
                            ),
                            subarticle_number=(
                                subarticle_number
                            ),
                        )
                    ),

                    "law_title": (
                        self.law_title
                    ),

                    "type": "subarticle",

                    "article_number": (
                        current_article[
                            "article_number"
                        ]
                    ),

                    "article_reference": (
                        current_article[
                            "article_reference"
                        ]
                    ),

                    "subarticle_number": (
                        subarticle_number
                    ),

                    "subarticle_reference": (
                        f"تبصره "
                        f"{subarticle_number} "
                        f"ماده "
                        f"{current_article['article_number']}"
                    ),

                    "chapter_title": (
                        current_article[
                            "chapter_title"
                        ]
                    ),

                    "section_title": (
                        current_article[
                            "section_title"
                        ]
                    ),

                    "text": (
                        subarticle_text
                    ),

                    "source_url": (
                        self.source_url
                    ),
                }


                records.append(
                    current_subarticle
                )

                continue


            # =================================================
            # Continuation Text
            # =================================================

            if current_subarticle is not None:

                current_subarticle[
                    "text"
                ] = self._join_text(
                    current_subarticle[
                        "text"
                    ],
                    text,
                )


            elif current_article is not None:

                current_article[
                    "text"
                ] = self._join_text(
                    current_article[
                        "text"
                    ],
                    text,
                )


        return records


    # =========================================================
    # HTML Extraction
    # =========================================================

    def _extract_text_elements(
        self,
    ):
        """
        Extract text elements from the HTML source.

        This method follows the same extraction approach
        used by the original working parser.
        """

        elements = self.soup.find_all(
            [
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "p",
                "div",
                "li",
            ]
        )

        return [
            element
            for element in elements
            if self._clean_text(
                element.get_text(
                    " ",
                    strip=True,
                )
            )
        ]


    # =========================================================
    # Text Normalization
    # =========================================================

    @staticmethod
    def _clean_text(
        text: str,
    ) -> str:
        """
        Normalize whitespace while preserving Persian text.
        """

        text = text.replace(
            "\xa0",
            " ",
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()


    @staticmethod
    def _join_text(
        previous: str,
        new: str,
    ) -> str:
        """
        Append continuation text.
        """

        if not previous:

            return new


        if not new:

            return previous


        return (
            f"{previous}\n{new}"
        )


    # =========================================================
    # Subarticle Helpers
    # =========================================================

    @staticmethod
    def _get_next_subarticle_number(
        records: list[dict],
        article_number: int,
    ) -> int:
        """
        Generate the next sequential subarticle number
        for an article.

        Example:

            Article 27
                Note -> 1
                Note -> 2
                Note -> 3
        """

        existing_numbers = [

            record[
                "subarticle_number"
            ]

            for record in records

            if (
                record.get(
                    "type"
                )
                == "subarticle"
                and record.get(
                    "article_number"
                )
                == article_number
            )
        ]


        if not existing_numbers:

            return 1


        return max(
            existing_numbers
        ) + 1


    @staticmethod
    def _build_subarticle_id(
        article_number: int,
        subarticle_number: int,
    ) -> str:
        """
        Build a stable unique ID.
        """

        return (
            f"article-{article_number}"
            f"-subarticle-{subarticle_number}"
        )


    # =========================================================
    # Validation
    # =========================================================

    @staticmethod
    def validate(
        records: list[dict],
    ) -> dict:
        """
        Validate parsed records.

        Validation is centralized here so the notebook
        does not contain many separate validation cells.
        """

        articles = [

            record

            for record in records

            if record.get(
                "type"
            )
            == "article"
        ]


        subarticles = [

            record

            for record in records

            if record.get(
                "type"
            )
            == "subarticle"
        ]


        invalid_articles = [

            record

            for record in articles

            if not record.get(
                "text",
                "",
            ).strip()
        ]


        invalid_subarticles = [

            record

            for record in subarticles

            if (
                not record.get(
                    "text",
                    "",
                ).strip()
            )

            or (
                record.get(
                    "article_number"
                )
                is None
            )
        ]


        article_numbers = [

            record.get(
                "article_number"
            )

            for record in articles
        ]


        report = {

            "status": "PASSED",

            "total_records": len(
                records
            ),

            "article_count": len(
                articles
            ),

            "subarticle_count": len(
                subarticles
            ),

            "unique_article_count": len(
                set(
                    article_numbers
                )
            ),

            "invalid_articles": len(
                invalid_articles
            ),

            "invalid_subarticles": len(
                invalid_subarticles
            ),
        }


        if (

            report[
                "total_records"
            ]
            == 0

            or report[
                "invalid_articles"
            ]
            > 0

            or report[
                "invalid_subarticles"
            ]
            > 0
        ):

            report[
                "status"
            ] = "FAILED"


        return report


    # =========================================================
    # Save
    # =========================================================

    @staticmethod
    def save(
        records: list[dict],
        output_path: Path,
        law_title: str = "قانون کار",
        source_url: str | None = None,
    ) -> None:
        """
        Save parsed records as JSON.
        """

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        validation = (
            LaborLawParser.validate(
                records
            )
        )


        output = {

            "metadata": {

                "law_title": (
                    law_title
                ),

                "source_url": (
                    source_url
                ),

                "record_count": (
                    len(records)
                ),

                "article_count": (
                    validation[
                        "article_count"
                    ]
                ),

                "subarticle_count": (
                    validation[
                        "subarticle_count"
                    ]
                ),

                "validation_status": (
                    validation[
                        "status"
                    ]
                )
            },

            "records": records,
        }


        output_path.write_text(

            json.dumps(
                output,
                ensure_ascii=False,
                indent=2,
            ),

            encoding="utf-8",
        )