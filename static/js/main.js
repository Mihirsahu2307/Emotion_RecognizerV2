$(document).ready(function(){
    var socket = io()
    // var socket = io.connect(window.location.protocol + '//' + document.domain + ':' + location.port);
    socket.on('connect', function(){
        console.log("Connected...!", socket.connected)
    });

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
          
            // Next, we can send the FormData object to the Python server using an HTTP POST request
            const xhr = new XMLHttpRequest();
            xhr.open('POST', '/webcam', true);
            xhr.send(formData);
        }, 'image/jpeg', 1.0);

        context.clearRect(0, 0, width,height );
    }


    socket.on('response_back', function(image){
            // console.log('You got me!')
            photo.setAttribute('src', image );
    });
});