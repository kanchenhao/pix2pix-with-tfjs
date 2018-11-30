/* variables */

var canvas,
    currColor = '#002FFF',
    backColor = '#ffffff',
    gCanvas = document.getElementById('gCanvas'),
    isRedoing = false,
    h = [],
    model = undefined,
    imgData_model = new Image;

imgData_model.src = './img/demo.jpg';

/* color pallette click events */
$(document).on("click", "td", function (e) {
    const color = e.target.style.backgroundColor;
    currColor = color;
});

/* prepare the drawing canvas  */
function prepareCanvas() {
    canvas_model = window._canvas = new fabric.Canvas('canvas-model');
    fabric.Image.fromURL('./img/demo.jpg', function (oImg) {
        canvas_model.add(oImg);
        canvas_model.item(0).selectable = false;
    });
    canvas = window._canvas = new fabric.Canvas('canvas');
    fabric.Image.fromURL('./img/201d.jpg', function (oImg) {
        canvas.add(oImg);
    });
    gCanvas = window._canvas = new fabric.Canvas('gCanvas');
    // fabric.Image.fromURL('./img/201b.jpg', function (oImg) {
    //     gCanvas.add(oImg);
    //     gCanvas.item(0).selectable = false;
    // });
    canvas.backgroundColor = '#ffffff';
    canvas.isDrawingMode = 1;
    canvas.freeDrawingBrush.color = 'rgb(172,172,172)';
    canvas.freeDrawingBrush.width = 2;
    canvas.renderAll();
    //setup listeners 
    canvas.on('mouse:up', function (e) {
        /*
        const imgData = getImageData();
        const pred = predict(imgData)
        tf.toPixels(pred, gCanvas)
        */
        mousePressed = false
    });
    canvas.on('mouse:down', function (e) {
        mousePressed = true
    });
    canvas.on('object:added', function () {
        if (!isRedoing) {
            h = [];
        }
        isRedoing = false;
    });
}

/* load the model */
async function start(imgName, MODEL_PATH, WEIGHTS_PATH) {
    //load the model
    // model = await tf.loadModel(MODEL_PATH);
    model = await tf.loadFrozenModel(MODEL_PATH, WEIGHTS_PATH);
    populateInitImage(imgName).then(function (result) {
        document.getElementById('status').innerHTML = 'Model Loaded';
    });

    // allowDrawing();
    /* allow drawing on canvas */
    canvas.isDrawingMode = 1;
    $('#clear').prop('disabled', false);
}

/* predict on initial image */
function populateInitImage(imgName) {
    var imgData = new Image;
    imgData.src = imgName;

    document.getElementById('status').innerHTML = 'Model Initing';
    document.getElementById('bar').style.display = "none"
    var Promise1 = new Promise(function (resolve) {
        imgData.onload = function () {
            const img = new fabric.Image(imgData, {
                scaleX: canvas.width / 512,
                scaleY: canvas.height / 512,
            });
            // const img = new fabric.Image(imgData);
            canvas.add(img)
            const img_model = new fabric.Image(imgData_model);
            canvas_model.add(img_model)
            const pred = predict(imgData_model, imgData)
            tf.toPixels(pred, gCanvas)
            resolve();
        }
    })
    return Promise1;
}

/* get the prediction */
function predict(imgData_model, imgData) {
    return tf.tidy(() => {
        //get the prediction 
        const gImg = model.predict(preprocess(imgData_model, imgData))
        //post process
        const postImg = postprocess(gImg)
        return postImg
    })
}

/* preprocess the data */
function preprocess(imgData_model, imgData) {
    return tf.tidy(() => {
        //convert to a tensor 
        const tensor = tf.fromPixels(imgData).toFloat();
        //resize
        const resized = tf.image.resizeBilinear(tensor, [512, 512]);
        //normalize
        const offset = tf.scalar(127.5);

        const normalized = resized.div(offset).sub(tf.scalar(1.0));
        //We add a dimension to get a batch shape 
        const batched = normalized.expandDims(0);

        const tensor_model = tf.fromPixels(imgData_model).toFloat();
        const resized_model = tf.image.resizeBilinear(tensor_model, [512, 512]);
        const normalized_model = resized_model.div(offset).sub(tf.scalar(1.0));
        const batched_model = normalized_model.expandDims(0);
        const batched_end = tf.concat([batched_model, batched], -1)
        return batched_end
    })

}

/* post process */
function postprocess(tensor) {
    const w = canvas.width
    const h = canvas.height
    return tf.tidy(() => {
        //normalization factor  
        const scale = tf.scalar(0.5);
        //unnormalize and sqeeze 
        const squeezed = tensor.squeeze().mul(scale).add(scale)
        //resize to canvas size 
        const resized = tf.image.resizeBilinear(squeezed, [w, h])
        return resized
    })
}

/* processing */
function processing() {
    document.getElementById('process').innerHTML = 'running';
    const imgData = getImageData();
    const pred = predict(imgData_model, imgData)
    tf.toPixels(pred, gCanvas)
    document.getElementById('process').innerHTML = 'process'
}


/* get the current image data */
function getImageData() {
    //get image data according to dpi 
    const dpi = window.devicePixelRatio
    const x = 0 * dpi
    const y = 0 * dpi
    const w = canvas.width * dpi
    const h = canvas.height * dpi
    const imgData = canvas.contextContainer.getImageData(x, y, w, h)
    return imgData
}

/* release resources when leaving the current page */
function release() {
    if (model != undefined) {
        model.dispose()
    }
}

window.onbeforeunload = function (e) {
    console.log('leaving the page')
    release()
}
$('.nav-link').click(function () {
    release()
})

//undo
function undo() {
    if (canvas._objects.length > 0) {
        h.push(canvas._objects.pop());
        canvas.renderAll();
    }
}

//redo
function redo() {
    if (h.length > 0) {
        isRedoing = true;
        canvas.add(h.pop());
    }
}

/* clear the canvas */
function clear_all() {
    canvas.clear();
    gCanvas.clear();
    gCanvas.backgroundColor = backColor;
    canvas.backgroundColor = backColor;
}

//init
function init() {
    start("./img/201d.jpg", "./model/tensorflowjs_model.pb", "./model/weights_manifest.json");
}

/* save model figure */
function save() {
    var data_b64 = $("#canvas").get(0).toDataURL("image/png").replace(/^data:image\/png;base64,/, "")
    var data = b64_to_bin(data_b64)
    var blob = new Blob([data], {
        type: "application/octet-stream"
    })
    var url = window.URL.createObjectURL(blob)
    var a = document.createElement("a")
    a.href = url
    a.download = "pix2pix.png"
    a.click()
    // use createEvent instead of .click() to work in firefox
    // also can"t revoke the object url because firefox breaks
    // var event = document.createEvent("MouseEvents")
    // event.initEvent("click", true, true)
    // a.dispatchEvent(event)

    function b64_to_bin(str) {
        var binstr = atob(str)
        var bin = new Uint8Array(binstr.length)
        for (var i = 0; i < binstr.length; i++) {
            bin[i] = binstr.charCodeAt(i)
        }
        return bin
    }
    
}