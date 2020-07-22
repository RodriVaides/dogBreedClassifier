//Bucket Configurations
var bucketName = "NAME OF THE BUCKET WHERE YOU WILL UPLOAD THE FILES"; // Fill in with your bucket name
var bucketRegion = "eu-central-1";
var IdentityPoolId = "IDENTITY POOL ID"; // Fill in with the ID of the Identity Pool created

 AWS.config.update({
                region: bucketRegion,
                credentials: new AWS.CognitoIdentityCredentials({
                    IdentityPoolId: IdentityPoolId
                })
            });

            var s3 = new AWS.S3({
                apiVersion: '2006-03-01',
                params: {Bucket: bucketName}
        });
		

// Upload function
function s3upload() {
   var files = document.getElementById('real-file').files;
   if (files) 
   {
     var file = files[0];
     var fileName = file.name;
     var filePath = 'img_inputs/' + fileName;
     var fileUrl = 'https://' + bucketName + '.s3.' + bucketRegion + '.amazonaws.com/' +  filePath;
	 var review = document.getElementById('review')
	 review.innerHTML = fileUrl
	 // alert(fileUrl)
     s3.upload({
        Key: filePath,
        Body: file,
        ACL: 'public-read'
        }, function(err, data) {
        if(err) {
        reject('error');
        }
        alert('Successfully Uploaded!');
        }).on('httpUploadProgress', function (progress) {
        var uploaded = parseInt((progress.loaded * 100) / progress.total);
        $("progress").attr('value', uploaded);
      });
   }
};