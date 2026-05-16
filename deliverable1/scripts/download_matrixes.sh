# Create folder and if it exists, clean it
mkdir -p ./data
rm -rf ./data
mkdir -p ./data
urls=(
    "https://suitesparse-collection-website.herokuapp.com/MM/Freescale/FullChip.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/PARSEC/Ga41As41H72.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/PARSEC/Si41Ge41H72.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Oberwolfach/bone010.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/ldoor.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Rajat/rajat31.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Sandia/ASIC_680ks.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Rucci/Rucci1.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/GHS_indef/boyd2.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Williams/webbase-1M.tar.gz"
)

for url in "${urls[@]}"; do
    tmpdir=$(mktemp -d)
    curl -L --insecure "$url" | tar -xz -C "$tmpdir"
    find "$tmpdir" -name "*.mtx" -exec mv {} ./data/ \;
    rm -rf "$tmpdir"
done

./sort_matrixes.sh