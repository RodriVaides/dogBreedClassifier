//Initializing strict mode
 "use strict"; 
 
 // Function used to submit to sagemaker the URL of the image uploaded to S3
 function submitForm(oFormElement) {
     var xhr = new XMLHttpRequest();
     xhr.onload = function() {
		 
         var result_raw = xhr.responseText;
		 var result = JSON.parse(result_raw.replace(/'/g,'"'));	
		 console.log(result)
         var resultElement = document.getElementById('result');
		 resultElement.innerHTML = "Your result is: ".concat(result.predicted_name);
		 var resultImg = document.getElementById('result_img');
		 resultImg.src = result.predicted_result_url
		 resultImg.style.visibility="visible"
		 var inputImg = document.getElementById('input_img');
		 inputImg.src = result.input_img_url
		 inputImg.style.visibility="visible"

     }
	 xhr.onerror= function() {
	     alert(" ¯\\_(ツ)_/¯\n The application is currently not running, please contact me to start it (so that I don't get charged by Amazon when it's not being used :D)");
	 };
     xhr.open (oFormElement.method, oFormElement.action, true);
     var inputUrl = document.getElementById('inputUrl');
     xhr.send (inputUrl.value);
     return false;
 }