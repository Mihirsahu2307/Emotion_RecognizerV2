$(document).ready(function(){
    var canvas = document.getElementById('canvas');
    var context = canvas.getContext('2d');
    const video = document.querySelector("#videoElement");
    const form = document.getElementById('my-form');
    const but = document.getElementById('SendBtn')

    form.addEventListener('submit', function(event) {
        event.preventDefault();

        submitFrame();

        but.disabled = true;
        setTimeout(function(){
            but.disabled = false;
        }, 1500);
    });

    video.width = 640;
    video.height = 480; 


    if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
        .then(function (stream) {
            video.srcObject = stream;
            video.play();
        })
        .catch(function (err0r) {

        });
    }

    function submitFrame(){
        width=video.width;
        height=video.height;
        context.drawImage(video, 0, 0, width , height );

        canvas.toBlob(function(blob) {
            // Now, we can use the Blob object to create a FormData object
            const formData = new FormData();
            formData.append('image', blob);
          
            fetch("/webcam", {
                method: 'POST',
                body: formData,
                //headers: {"Content-type": "application/x-www-form-urlencoded; charset=UTF-8"}  // makes only problem
            }).then(function(response) {
                return response.blob();
            }).then(function(blob) {
                //console.log(blob);  // it slow down video from server
                console.log('Heh, you got me!')
                photo.src = URL.createObjectURL(blob);
            }).catch(function(err) {
                console.log('Fetch problem: ' + err.message);
            });
        
            // $.ajax({
            //     type: 'POST',
            //     url: '/webcam',
            //     data: formData,
            //     contentType: false,
            //     cache: false,
            //     processData: false,
            //     success: function(data) {
            //         console.log('Great Success!');
            //     },
            // });

            console.log('Adios')
        }, 'image/jpeg', 1.0);

        context.clearRect(0, 0, width,height );
    }


    // socket.on('response_back', function(image){
    //         console.log('You got me!')
    //         photo.setAttribute('src', image );
    // });
});