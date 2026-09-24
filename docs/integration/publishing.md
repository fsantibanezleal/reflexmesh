# First publication without sharing credentials

The package is published through GitHub's OIDC identity. No PyPI password or long-lived API token is needed by the assistant, the repository, or its workflow.

Use your normal browser with your existing PyPI login. Open https://pypi.org/manage/account/publishing/ and add a new pending GitHub publisher:

| Field | Value |
| --- | --- |
| PyPI project name | reflexmesh |
| GitHub owner | fsantibanezleal |
| Repository name | reflexmesh |
| Workflow filename | publish-pypi.yml |
| Environment name | pypi |

Click Add. The first successful workflow publication creates the project. A pending publisher does not reserve the name. See the [official PyPI procedure](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

The GitHub environment restricts deployments to main and version tags. The release workflow builds and checks portable distributions, then exchanges its short-lived GitHub identity for PyPI publication authority. Release uploads include attestations. Package publication, model asset upload and public-site deployment are separate checks.
