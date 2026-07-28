from pathlib import Path

import requests

def fetch_html(
    url: str,
    output_path: Path | None = None,
    timeout: int = 30,
) -> str:
    """
    Download HTML content from the source URL.

    Parameters
    ----------
    url:
        Source URL.

    output_path:
        Optional path to save raw HTML.

    timeout:
        Request timeout in seconds.

    Returns
    -------
    str
        Raw HTML content.
    """

    headers = {

        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/150.0 Safari/537.36"
        )
    }


    response = requests.get(

        url,

        headers=headers,

        timeout=timeout,
    )


    response.raise_for_status()


    response.encoding = (

        response.apparent_encoding

        or response.encoding

    )


    html = response.text


    if output_path is not None:

        output_path.parent.mkdir(

            parents=True,

            exist_ok=True,
        )


        output_path.write_text(

            html,

            encoding="utf-8",
        )


    return html