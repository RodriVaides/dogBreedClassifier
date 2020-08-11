// ---------REFERENCES-----------
// Reference for customizing file upload custom_file_upload_button
// https://www.youtube.com/watch?v=T3PDgtliezo

const realFileBtn = document.getElementById("real-file");
const customBtn = document.getElementById("custom-button");
const customTxt = document.getElementById("custom-text");

customBtn.addEventListener("click", function() {
	realFileBtn.click();
});

realFileBtn.addEventListener("change", function(){
	// alert(realFileBtn.files[0].name)
	if (realFileBtn.value){
		customTxt.innerHTML = realFileBtn.files[0].name;
	} else {
		customTxt.innerHtml = "No file chosen, yet.";
	}
})
