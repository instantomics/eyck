# uv add setuptools_git_versioning gh twine build
python -m setuptools_git_versioning

version="0.1.0"

git add .
git commit -m "version v${version}"

git tag -a v${version} -m "v${version}"

python -m build

# optional: upload test
# twine upload --repository testpypi dist/eyck-${version}.tar.gz --verbose

git push --tags
git push

# curl -sS https://webi.sh/gh | sh
~/.local/bin/gh release create v${version} -t "v${version}" -n "v${version}" dist/eyck-${version}.tar.gz

# pip install twine
twine upload dist/eyck-${version}.tar.gz --verbose

python -m build --wheel
