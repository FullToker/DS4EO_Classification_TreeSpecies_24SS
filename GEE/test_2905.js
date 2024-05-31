var Boundary = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017");


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

/***Select the roi of Germany***/
Map.setCenter(16.35, 48.83, 4);
var dataset = Boundary.select("country_co");
var roi_germany = dataset.filter(ee.Filter.eq('country_co','GM'))
Map.addLayer(roi_germany, null, 'roi of Germany')


/***Select Sentinel-2 data***/
var s2_spring = ee.ImageCollection('COPERNICUS/S2_SR')
      .filterBounds(roi_germany)
      .filterDate('2022-03-01', '2022-05-31')
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))
      .map(function(image){
            return image.clip(roi_germany)
        })
      .map(maskS2clouds)
      .median();
      
var s2_summer = ee.ImageCollection('COPERNICUS/S2_SR')
      .filterBounds(roi_germany)
      .filterDate('2022-06-01', '2022-08-31')
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))
      .map(function(image){
            return image.clip(roi_germany)
        })
      .map(maskS2clouds)
      .median();
      
var s2_autumn = ee.ImageCollection('COPERNICUS/S2_SR')
      .filterBounds(roi_germany)
      .filterDate('2022-09-01', '2022-11-30')
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))
      .map(function(image){
            return image.clip(roi_germany)
        })
      .map(maskS2clouds)
      .median();
      
var s2 = s2_spring.addBands(s2_summer).addBands(s2_autumn);

var Bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'B2_1', 'B3_1', 'B4_1', 'B5_1', 'B6_1', 'B7_1', 'B8_1', 'B8A_1', 'B11_1', 'B12_1',
'B2_2', 'B3_2', 'B4_2', 'B5_2', 'B6_2', 'B7_2', 'B8_2', 'B8A_2', 'B11_2', 'B12_2']
var s2_select = s2.select(Bands)



/***Use forest mask to fliter samples***/
// Use COPERNICUS Land cover product as forest mask
var forest_mask = ee.Image("COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019")
.select('discrete_classification');

var forest = forest_mask.reduceRegions({
  collection: samples,
  reducer: ee.Reducer.first(),
  scale: 10,
  crs: 'EPSG:3035'
});

// Take "Sorbus_aucuparia" species as example
// Code 111 means closed needleleaf forest and 114 means closed broadleaf forest
var Sorbus_aucuparia = forest.filterMetadata('SPECIES NAME','equals','Sorbus aucuparia');
var Sorbus_aucuparia = Sorbus_aucuparia.filterMetadata('first','equals', 114);
/***Visualize species samples***/
Map.addLayer(Sorbus_aucuparia, {color: '#f3ff3c'}, 'Sorbus aucuparia');


/***Data export***/
// Set the patch size to 5 pixels
var kernel = ee.Kernel.rectangle(12, 12)
var neighborImg = s2_select.neighborhoodToArray(kernel)
var day = "0529"

// Clip the patch
var species = "Sorbus aucuparia";
var samples = neighborImg.reduceRegions({
  collection:  Sorbus_aucuparia,
  reducer: ee.Reducer.first(),
  scale: 10,
  crs: 'EPSG:3035'
});

// Export to Google Drive
Export.table.toDrive({
    collection : samples,
    description : species + '_' + day,
    fileNamePrefix : species + '_' + day,
    fileFormat : 'GeoJSON',
    selectors : Bands
  });