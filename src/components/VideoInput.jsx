function VideoUploader({ onVideo }) {

  const handleChange = (e) => {
    const file = e.target.files[0];

    if(file){
      onVideo(URL.createObjectURL(file));
    }
  };


  return (
    <input
      type="file"
      accept="video/*"
      onChange={handleChange}
    />
  );
}

export default VideoUploader;