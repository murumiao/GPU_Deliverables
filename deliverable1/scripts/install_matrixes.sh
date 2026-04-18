# Create folder and if it exists, clean it
mkdir -p ./data
rm -rf ./data
mkdir -p ./data
urls=(
    "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/soc-sign-Slashdot081106.tar.gz"


    #undirected graph
    # "https://suitesparse-collection-website.herokuapp.com/MM/Mycielski/mycielskian19.tar.gz"

    #2d/3d problem | solo diagonale (ish)
    # "https://suitesparse-collection-website.herokuapp.com/MM/Janna/Queen_4147.tar.gz"
)

for url in "${urls[@]}"; do
    tmpdir=$(mktemp -d)
    curl -L --insecure "$url" | tar -xz -C "$tmpdir"
    find "$tmpdir" -name "*.mtx" -exec mv {} ./data/ \;
    rm -rf "$tmpdir"
done
