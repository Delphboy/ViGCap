from typing import List

import requests


def download_from_url(url, path):
    """Download file, with logic (from tensor2tensor) for Google Drive"""
    if "drive.google.com" not in url:
        print("Downloading %s; may take a few minutes" % url)
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        with open(path, "wb") as file:
            file.write(r.content)
        return
    print("Downloading from Google Drive; may take a few minutes")
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v

    if confirm_token:
        url = url + "&confirm=" + confirm_token
        response = session.get(url, stream=True)

    chunk_size = 16 * 1024
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)


def preprocess_captions(captions: List[str]) -> List[str]:
    # Clean sentence list following: https://cs.stanford.edu/people/karpathy/cvpr2015.pdf Section 4
    captions = [caption.lower() for caption in captions]

    # Disgard non-alphanumeric characters
    non_alphanumeric = [chr(i) for i in range(33, 128) if not chr(i).isalnum()]
    cleaned = []

    for sentence in captions:
        for char in non_alphanumeric:
            sentence = sentence.replace(char, "")
        cleaned.append(sentence.strip())
        while "  " in sentence:
            sentence = sentence.replace("  ", " ")
    return cleaned
