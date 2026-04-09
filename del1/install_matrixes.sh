
mkdir -p ./data

urls=(
    "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/soc-sign-Slashdot081106.tar.gz"
)

for url in "${urls[@]}"; do
    tmpdir=$(mktemp -d)
    curl -L --insecure "$url" | tar -xz -C "$tmpdir"

    find "$tmpdir" -name "*.mtx" -exec mv {} ./data/ \;
    rm -rf "$tmpdir"
done