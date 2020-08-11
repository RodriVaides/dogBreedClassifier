
// Reference project for code used to upload files to Amazon S3
// https://medium.com/@shresthshruti09/uploading-files-in-aws-s3-bucket-through-javascript-sdk-with-progress-bar-d2a4b3ee77b5
// https://docs.aws.amazon.com/sdk-for-javascript/v2/developer-guide/s3-example-photo-album.html

//Bucket Configurations
var bucketName = "my-sage-maker-instance-test-20-03-2020-2";
var bucketRegion = "eu-central-1";
var IdentityPoolId = "eu-central-1:83237b44-78f2-494e-b25d-519239335ded";

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
	 var inputUrl = document.getElementById('inputUrl')
	 inputUrl.innerHTML = fileUrl
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
