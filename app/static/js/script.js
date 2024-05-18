$(document).ready(function(){
  var dataTransfer = new DataTransfer();
  $('#filess').change(function(e){
    var filelength =this.files.length;
    if(filelength==0) {
      $('.preview .image').children().remove();
    }
    if(filelength>0) {
      $('.preview .image').empty();
      for(let i=0;i<filelength;i++) {
        var fileName = window.URL.createObjectURL(this.files[i]);
        var uploadedFileName = this.files[i].name;
        $('.preview .image').append('<div class="images mt-3"><img class="my_image" src="'+fileName+'" alt="" height="200px" width="200px"><span class="close"><i class="fa-solid fa-xmark"></i></span><h4 class="file-name">'+uploadedFileName+'</h4></div>');
        // Add the file to the DataTransfer object
        dataTransfer.items.add(this.files[i]);
      }
      // Set the files property of the file input
      $('#filess')[0].files = dataTransfer.files;

    }
  });
  $(document).on('click','.close',function(){
    var indeX = $(this).parent().index();
    var filename = $(this).next('h4').text();

    // Validate the filename against the DataTransfer object
    if (filename === dataTransfer.items[indeX].getAsFile().name) {
      // Remove the specific image preview from the DOM
      $(this).parent().remove();

      // Remove the file from the DataTransfer object
      dataTransfer.items.remove(indeX);

      // Set the files property of the file input
      $('#filess')[0].files = dataTransfer.files;
    }
  })
});