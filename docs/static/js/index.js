window.HELP_IMPROVE_VIDEOJS = false;

// var INTERP_BASE = "./static/interpolation/stacked";
// var NUM_INTERP_FRAMES = 240;

// var interp_images = [];
// function preloadInterpolationImages() {
//   for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
//     var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
//     interp_images[i] = new Image();
//     interp_images[i].src = path;
//   }
// }

// function setInterpolationImage(i) {
//   var image = interp_images[i];
//   image.ondragstart = function() { return false; };
//   image.oncontextmenu = function() { return false; };
//   $('#interpolation-image-wrapper').empty().append(image);
// }


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    // preloadInterpolationImages();

    // $('#interpolation-slider').on('input', function(event) {
    //   setInterpolationImage(this.value);
    // });
    // setInterpolationImage(0);
    // $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();

    // Demo logic
    var filenameSelector = document.getElementById("filename");

    filenameSelector.addEventListener("change", function() {
      let input_path = "./static/demo/input/" + filenameSelector.value + ".png";
      let opr_path = "./static/demo/opr/" + filenameSelector.value + ".png";
      let excoltran_path = "./static/demo/excoltran+opr/content_" + filenameSelector.value + "_fake_R_rgb_3.png";
      let rehistogan_path = "./static/demo/rehistogan+opr/output_" + filenameSelector.value + "_generated.png";
      let mast_path = "./static/demo/mast+opr/" + filenameSelector.value + "_" + filenameSelector.value + ".png";
      let pcapst_path = "./static/demo/pcapst+opr/" + filenameSelector.value + ".png";
      let ours_path = "./static/demo/ours/" + filenameSelector.value + ".png";

      document.getElementById("demo_input").src = input_path;
      document.getElementById("demo_opr").src = opr_path;
      document.getElementById("demo_excoltran").src = excoltran_path;
      document.getElementById("demo_rehistogan").src = rehistogan_path;
      document.getElementById("demo_mast").src = mast_path;
      document.getElementById("demo_pcapst").src = pcapst_path;
      document.getElementById("demo_ours").src = ours_path;
    });
    
})


