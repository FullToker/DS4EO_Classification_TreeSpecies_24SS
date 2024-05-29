var S2A = ee.ImageCollection("COPERNICUS/S2_SR"); //引入sentinel-2 MSI,L2A影像

var Boundary = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017");
Map.setCenter(16.35, 48.83, 4);
var dataset = Boundary.select("country_co");
var roi_germany = dataset.filter(ee.Filter.eq('country_co','GM'))

var roi = ee.FeatureCollection("users/equalfree12738/ROI_testshp")  //


/***Remove clouds***/
function maskS2clouds(image) {
  var qa = image.select('QA60');
 
  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
 
  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
 
  return image.updateMask(mask).divide(10000);
}


var S2AImgs = S2A.filterBounds(roi_germany) 
               .filterDate("2022-03-01", "2022-05-31")
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',50));
               //.map(maskS2clouds);//筛选+去云

var s2_spring = S2A
      .filterBounds(roi_germany)
      .filterDate('2022-03-01', '2022-05-31')
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))
      .map(function(image){
            return image.clip(roi_germany)
        })
      .map(maskS2clouds)
      .median();

//print("sentinel-2_Images", S2AImgs); 
Map.addLayer(s2_spring, {min:0, max:0.3, bands:["B4","B3","B2"]}, "sentinel-2_Images"); 
Map.addLayer(roi_germany, {color: "red"}, "roi"); 
//var roi2=S2AImgs.first().geometry();
//Map.addLayer(roi2)

//以上代码已经完全改完

//影像集合导出方法(toDrive)
var indexList = S2AImgs.reduceColumns(ee.Reducer.toList(), ["system:index"]) 
                        .get("list"); 
  indexList.evaluate(function(indexs) { 
    for (var i=0; i<indexs.length; i++) { 
      var image = S2AImgs.filter(ee.Filter.eq("system:index", indexs[i])).first(); 
      image = image; 
      Export.image.toDrive({
        image: image.int16(),
        description: indexs[i], 
        fileNamePrefix: indexs[i], 
        region: roi, 
        scale: 10, 
        crs: "EPSG:4326", 
        maxPixels: 1e13 
      }); 
    } 
  }); 