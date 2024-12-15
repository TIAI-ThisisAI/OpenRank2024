import requests
import csv
import time


def fetch_citations(paper_id, fields=None, limit=1000):
    """
    Fetch all citations for a given paper using Semantic Scholar API.

    Args:
        paper_id (str): The paperId for which to fetch citations.
        fields (str, optional): A comma-separated list of fields to include in the response.
        limit (int): Maximum number of citations per API request (default is 100, max is 1000).

    Returns:
        list: A list of dictionaries containing citation information.
    """
    base_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"
    citations = []
    offset = 0

    while True:
        params = {"offset": offset, "limit": limit}
        if fields:
            params["fields"] = fields

        try:
            response = requests.get(base_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                citations.extend(data.get("data", []))

                # Check if there are more results
                if "next" in data and data["next"] is not None:
                    offset = data["next"]
                else:
                    break
            else:
                print(f"Error: Status code {response.status_code} - {response.text}")
                break
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            break

        # Delay to avoid rate-limiting
        time.sleep(1)

    return citations


def save_citations_to_csv(citations, filename):
    """
    Save citation information to a CSV file.

    Args:
        citations (list): A list of citation data.
        filename (str): The name of the CSV file to save.
    """
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["Paper Title", "Authors", "Publication Year"])

        for citation in citations:
            citing_paper = citation.get("citingPaper", {})
            title = citing_paper.get("title", "Unknown")
            year = citing_paper.get("year", "Unknown")
            authors = ", ".join([author.get("name", "Unknown") for author in citing_paper.get("authors", [])])

            writer.writerow([title, authors, year])


# Main program
if __name__ == "__main__":
    paper_id = "451d4a16e425ecbf38c4b1cca0dcf5d9bec8255c"  # Replace with your paperId
    fields = "citingPaper.title,citingPaper.authors,citingPaper.year"
    output_file = "paper/Image-large-Gradient-based learning applied to document recognition.csv"

    # Fetch citations
    print("Fetching citations...")
    citations = fetch_citations(paper_id, fields=fields)

    # Save to CSV
    print(f"Saving {len(citations)} citations to {output_file}...")
    save_citations_to_csv(citations, output_file)
    print("Done!")

# d2c733e34d48784a37d717fe43d9e93277a8c53e
# 3eda43078ae1f4741f09be08c4ecab6229046a5c NewsQA: A Machine Comprehension Dataset

# Image
# 162d958ff885f1462aeda91cd72582323fd6a1f4