import os
import stat
import uuid

from git import Repo
from common import config

# Rich
from rich.console import Console
console = Console()


def get_repository_url(url):
    if url.strip() == "":
        url = config.get('GITHUB_REPOSITORY')

    return url


def get_repository_name(url: str):
    return url.split("/")[-1].split(".")[0]


def get_repository(repository_url: str, clone: bool):
    git_username = config.get('GITHUB_USERNAME')
    git_access_token = config.get('GITHUB_ACCESS_TOKEN')

    if not "http" in repository_url:
        console.print(f"Invalid link to GitHub repository.", style="red")
        exit(1)

    split_url = repository_url.split("//")
    if git_username is not None and git_access_token is not None:
        auth_repository_url = f"//{git_username}:{git_access_token}@".join(
            split_url
        )
    else:
        auth_repository_url = "//".join(split_url)
    try:
        # repository_dir = os.path.join(get_repository_name(repository_url), uuid.uuid4().hex)
        repository_dir = get_repository_name(repository_url)
        repository_path = os.path.join(f".{os.path.sep}repositories", repository_dir)

        if clone:
            if os.path.exists(repository_path):
                remove_repository(repository_path)

            Repo.clone_from(
                url=auth_repository_url,
                to_path=repository_path
            )

            console.print(f"Cloned {repository_url} to {repository_path}", style="green")

        return repository_path
    except Exception as e:
        console.print(f"Error: {str(e)}", style="red")
        exit(1)


def remove_repository(directory: str):
    console.print(f"Removing directory: {directory}", style="yellow")
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            os.chmod(filename, stat.S_IWUSR)
            os.remove(filename)
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(directory)
