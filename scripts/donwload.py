from pathlib import Path
import time
import requests


def get_with_retries(url, headers=None, max_retries=5, timeout=60, stream=False):
    last_error = None

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout, stream=stream)

            if response.status_code in [429, 500, 502, 503, 504]:
                wait = 2 ** attempt
                print(f"Erro {response.status_code}. Tentando de novo em {wait}s...")
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response

        except requests.exceptions.RequestException as e:
            last_error = e
            wait = 2 ** attempt
            print(f"Erro: {e}. Tentando de novo em {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"Falha após {max_retries} tentativas: {url}") from last_error


def download_file(file_url, output_path, headers=None, max_retries=5):
    output_path = Path(output_path)

    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Arquivo já existe, pulando: {output_path}")
        return

    temp_path = output_path.with_suffix(output_path.suffix + ".part")

    for attempt in range(max_retries):
        try:
            with requests.get(file_url, headers=headers, stream=True, timeout=180) as r:
                if r.status_code in [429, 500, 502, 503, 504]:
                    wait = 2 ** attempt
                    print(f"Erro {r.status_code} ao baixar arquivo. Tentando de novo em {wait}s...")
                    time.sleep(wait)
                    continue

                r.raise_for_status()

                with open(temp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

            temp_path.rename(output_path)
            print("Salvo em:", output_path)
            return

        except requests.exceptions.RequestException as e:
            wait = 2 ** attempt
            print(f"Erro ao baixar {output_path.name}: {e}. Tentando de novo em {wait}s...")
            time.sleep(wait)

    print(f"Não consegui baixar: {output_path.name}")


def download_zenodo_record(record_id, output_dir="data/raw"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api_url = f"https://zenodo.org/api/records/{record_id}"

    headers = {
        "User-Agent": "privacy-ml-course-project/0.1"
    }

    response = get_with_retries(api_url, headers=headers, timeout=60)
    record = response.json()

    print("Title:", record["metadata"]["title"])

    for file_info in record["files"]:
        filename = file_info["key"]
        file_url = file_info["links"]["self"]
        output_path = output_dir / filename

        print(f"Baixando {filename}...")
        download_file(file_url, output_path, headers=headers)


if __name__ == "__main__":
    download_zenodo_record(record_id="3745945", output_dir="data/raw/pan15")