folder_path="$HOME/Desktop/Tesi/gamma/src/GAMMA"

# Navigate to the folder
cd "$folder_path" || exit

# Delete .csv files
find . -type f -name "*.csv" -delete

# Delete .m files
find . -type f -name "*.m" -delete

echo "Deletion complete for .csv and .m files in $folder_path"