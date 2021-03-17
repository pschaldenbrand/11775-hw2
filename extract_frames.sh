for video in videos/*.mp4
do
    base=$(basename -- "$video")
    base="${base%.*}"
    mkdir -p rgb/$base
    ffmpeg -i "$video"  rgb/"$base"/frame_%05d.jpg -hide_banner
done