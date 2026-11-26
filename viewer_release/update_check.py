"""Look for a newer Windows installer in the public GitHub releases."""

import json
import re
from dataclasses import dataclass
from http.client import HTTPException
from urllib.error import HTTPError
from urllib.request import Request, urlopen


try:
    from viewer_release._version import __version__
except ModuleNotFoundError as exc:
    if exc.name != 'viewer_release._version':
        raise
    __version__ = '0.0.0'

REPOSITORY = 'dubajicmilos/Hyperspectral-Viewer'
LATEST_RELEASE = f'https://api.github.com/repos/{REPOSITORY}/releases/latest'
VERSION_RE = re.compile(r'v?([0-9]{1,9})\.([0-9]{1,9})\.([0-9]{1,9})')


@dataclass(frozen=True)
class Update:
    version: str
    release_page: str


class UpdateCheckError(Exception):
    """The latest installer could not be determined."""


def _version_parts(value):
    matched = VERSION_RE.fullmatch(value) if isinstance(value, str) else None
    if matched is None:
        raise UpdateCheckError('The release version could not be read.')
    return tuple(int(part) for part in matched.groups())


def available_update(current_version=__version__):
    """Return an Update for a newer stable release with an uploaded setup file."""
    request = Request(LATEST_RELEASE, headers={
        'Accept': 'application/vnd.github+json',
        'User-Agent': f'TRPL-Viewer/{current_version}',
    })
    try:
        with urlopen(request, timeout=8) as response:
            release = json.loads(response.read(2 * 1024 * 1024))
    except HTTPError as exc:
        if exc.code == 404:
            raise UpdateCheckError('A public Windows release is not available yet.') from exc
        raise UpdateCheckError(f'GitHub returned HTTP {exc.code}. Try again later.') from exc
    except (HTTPException, OSError) as exc:
        raise UpdateCheckError('Could not reach GitHub. Check your connection and try again.') from exc
    except ValueError as exc:
        raise UpdateCheckError('GitHub returned an unreadable release.') from exc

    if not isinstance(release, dict) or release.get('draft') or release.get('prerelease'):
        raise UpdateCheckError('A stable release is not available yet.')
    tag = release.get('tag_name')
    latest = _version_parts(tag)
    if latest <= _version_parts(current_version):
        return None

    version = '.'.join(str(part) for part in latest)
    name = f'TRPL-Viewer-Setup-{version}-win-x64.exe'
    assets = release.get('assets')
    if not isinstance(assets, list) or not any(
        isinstance(asset, dict) and asset.get('name') == name and asset.get('state') == 'uploaded'
        for asset in assets
    ):
        raise UpdateCheckError('The latest release does not have a Windows installer yet.')

    return Update(version, f'https://github.com/{REPOSITORY}/releases/tag/{tag}')
