package com.example.mfs_tec

import android.annotation.SuppressLint
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.content.res.Configuration
import android.os.Build
import android.os.Bundle
import android.os.StrictMode
import android.telephony.SmsManager
import android.telephony.TelephonyManager
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.BottomNavigation
import androidx.compose.material.BottomNavigationItem
import androidx.compose.material.Checkbox
import androidx.compose.material.Divider
import androidx.compose.material.DropdownMenu
import androidx.compose.material.DropdownMenuItem
import androidx.compose.material.Icon
import androidx.compose.material.RadioButton
import androidx.compose.material.Switch
import androidx.compose.material.TextButton
import androidx.compose.material.TextField
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.AutoGraph
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.NotificationImportant
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.rememberDateRangePickerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.focus.FocusDirection
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat.getSystemService
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import co.yml.charts.axis.AxisData
import co.yml.charts.common.model.Point
import co.yml.charts.ui.linechart.LineChart
import co.yml.charts.ui.linechart.model.Line
import co.yml.charts.ui.linechart.model.LineChartData
import co.yml.charts.ui.linechart.model.LinePlotData
import co.yml.charts.ui.linechart.model.LineStyle
import co.yml.charts.ui.linechart.model.LineType
import co.yml.charts.ui.linechart.model.SelectionHighlightPoint
import co.yml.charts.ui.linechart.model.SelectionHighlightPopUp
import com.example.mfs_tec.ui.theme.MFSTECTheme
import com.google.android.material.datepicker.MaterialDatePicker.*
import com.microsoft.sqlserver.jdbc.*
import kotlinx.coroutines.*
import java.sql.*
import java.text.DecimalFormat
import java.time.LocalDateTime
import java.time.LocalTime
import java.time.format.DateTimeFormatter
import kotlin.math.floor

//General SQL variables
var dateValues = mutableListOf<String>()
var timeValues = mutableListOf<String>()
var SQL_error_message: MutableState<String> = mutableStateOf("")
//VDB Forecast variables
var VDBlocation = mutableListOf<String>()
var VDBforecast = mutableListOf<String>()
//ML Forecast variables
var MLforecast = mutableListOf<String>()
var MLlocation = mutableListOf<String>()
//Synthesis Forecast variables
var Synthforecast = mutableListOf<String>()
var Synthlocation = mutableListOf<String>()

//Chart labels
var VDBlabel: MutableState<String> = mutableStateOf("")
var MLlabel: MutableState<String> = mutableStateOf("")
var Synthlabel: MutableState<String> = mutableStateOf("")
var timelabel: MutableState<String> = mutableStateOf("")
var indexLabel: MutableState<Int> = mutableIntStateOf(0)

//Settings variables
var phoneNotification: MutableState<Boolean> = mutableStateOf(false)
var smsNotification: MutableState<Boolean> = mutableStateOf(false)
var regionValue: MutableState<String> = mutableStateOf("Coast")
var phoneNumber: MutableState<String> = mutableStateOf("Enter phone number")
var powerThreshold: MutableState<String> = mutableStateOf("Enter power threshold to be notified")
var dateAndTime = ""

val LimeGreen = Color(0xFFADD8E6)      //Color(0xFF00FF00) // Lime Green
val TealBlue = Color(0xFF008080) // Teal Blue
val Purple = Color(0xFF800080) // Purple

//Archive Variables
//var startDate: MutableState<String> = mutableStateOf("MM/DD/YYYY")
//var endDate: MutableState<String> = mutableStateOf("MM/DD/YYYY")

class MainActivity : ComponentActivity() {
    @SuppressLint("UnusedMaterial3ScaffoldPaddingParameter")
    //@OptIn(ExperimentalMaterial3Api::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        //Strict mode for SQL database connection
        val policy = StrictMode.ThreadPolicy.Builder().permitAll().build()
        StrictMode.setThreadPolicy(policy)
        //Variable for coroutine
        val coroutineScope = CoroutineScope(Dispatchers.Default)
        val applicationContext: Context = this


        var notificationTracker = true
        //Background notification checek
        coroutineScope.launch {
            //Loop if tracker is true and notification turned on in settings
            while (true) {
                //If notification already present, restart loop
                if (isDisplayingNotification(applicationContext)) {
                    delay(5 * 1000) //wait 5 seconds
                    Log.d("Notification Channel", "isDisplay is true")
                    continue
                }
                //Initialize to false
                var exists = false
                //Boolean variables for formatting
                val phoneFormat = phoneNumber.value.matches(Regex("[0-9]+"))
                val powerFormat = powerThreshold.value.matches(Regex("[0-9]+"))
                //If power is formatted, check for value
                if (powerFormat && (smsNotification.value || phoneNotification.value)) {
                    exists = checkValueExistsAsync(powerThreshold.value.toFloat())
                }
                val message = "Threshold of ${powerThreshold.value} MW exceeded " +
                        "in ${regionValue.value} region at $dateAndTime"
                Log.d("Notification System","Value exists: $exists")
                //Sends for phone notifications
                //Sends if value exists, format is correct, and notification tracker is true
                if (exists && phoneNotification.value && notificationTracker) {
                    showNotification(applicationContext, message)
                }
                //Sends for SMS notifications
                if (exists && smsNotification.value && phoneFormat && notificationTracker) {
                    sendText(phoneNumber.value, message)
                }
                delay(10 * 1000) // Delay for 10 seconds
            }
        }
        setContent {
            MFSTECTheme {
                MFS_tech_app()
            }
        }
    }
}

@SuppressLint("UnusedMaterial3ScaffoldPaddingParameter")
@Composable
fun MFS_tech_app() {
    val navController = rememberNavController()
    Scaffold(
        bottomBar = {
            BottomNavigationBar(navController)
        }
    ) {
        NavHost(navController, startDestination = Screen.Home.route) {
            composable(Screen.Home.route) { Home() }
            composable(Screen.Forecast.route) { Forecast() }
            //composable(Screen.Archive.route) { Archive() }
            composable(Screen.Settings.route) { Settings() }
        }
    }
}
sealed class Screen(val route: String) {
    object Home : Screen("home")
    object Forecast : Screen("forecast")
    object Archive : Screen("archive")
    object Settings : Screen("settings")
}
@Preview(widthDp = 380, heightDp = 800, backgroundColor = 1)
@Composable
fun Home() {
    val imagePainter = painterResource(id = R.drawable.ercot_map)
    val configuration = LocalConfiguration.current
    val isLandscape = configuration.orientation == Configuration.ORIENTATION_LANDSCAPE

    //Landscape Orientation
    if (isLandscape) {
        Row (
            modifier = Modifier.background(Color.White)
        ) {
            Box (
                modifier = Modifier
                    .width((GetScreenWidth() / 2 - 50).dp)
                    .height((GetScreenHeight() - 50).dp)
                    .padding(start = 60.dp)
            ) {
                Column (
                    modifier = Modifier.fillMaxSize(),
                    verticalArrangement = Arrangement.Center,
                    //horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "MFS-TECH",
                        modifier = Modifier.padding(horizontal = 16.dp),
                        color = Color.DarkGray,
                        fontSize = 34.sp,
                    )
                    Text(
                        text = "Multifactor Forecasting System for Texas Energy Consumption",
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
                        fontSize = 20.sp,
                        color = Color.Gray
                    )
                }
            }
            Image(
                painter = imagePainter,
                contentDescription = "ERCOT Region Map",
                modifier = Modifier
                    .height((GetScreenHeight() - 90).dp)
                    .padding(16.dp)
            )
            Spacer(modifier = Modifier.height(50.dp))
        }

    } else {
        //Portrait Orientation
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.White),
            verticalArrangement = Arrangement.Center,
            //horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = "MFS-TECH",
                modifier = Modifier.padding(horizontal = 16.dp),
                color = Color.DarkGray,
                fontSize = 34.sp,
            )
            Text(
                text = "Multifactor Forecasting System for Texas Energy Consumption",
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
                fontSize = 20.sp,
                color = Color.Gray
            )
            Spacer(modifier = Modifier.height(60.dp))
            Image(
                painter = imagePainter,
                contentDescription = "ERCOT Region Map",
                modifier = Modifier
                    .width((GetScreenWidth()).dp)
                    .padding(16.dp)
            )
            Spacer(modifier = Modifier.height(50.dp))

        }
    }



}

//@Preview(widthDp = 350, heightDp = 1000)
@Composable
fun Forecast() {
    //Format for refresh button time
    val formatter = DateTimeFormatter.ofPattern("MM/dd hh:mm a")
    val current = LocalDateTime.now().format(formatter)
    //Variables for data filtering
    var regionSelect = remember { mutableStateOf<String>("East") }
    var timeFrame = remember { mutableIntStateOf(336) }
        /*  1 Day: 24 hours
            3 Days: 72 hours
            7 days: 168 hours
            14 days: 336 hours  */
    var displayVDBforecast = remember { mutableStateOf<Boolean>(true)}
    var displayMLforecast = remember { mutableStateOf<Boolean>(false)}
    var displaySynthesis = remember { mutableStateOf<Boolean>(false)}
    //Filtered lists from the original
    var filteredTime: MutableList<String> = mutableListOf()
    var filteredDate: MutableList<String> = mutableListOf()
    var VDBPoints: MutableList<Point> = mutableListOf()
    var MLPoints: MutableList<Point> = mutableListOf()
    var SynthPoints: MutableList<Point> = mutableListOf()
    var refreshData by remember { mutableStateOf(false)}
    //Translate SQL labels to user friendly labels
    val locationRename = mapOf(
        "FAR_WEST" to "Far West",
        "NORTH" to "North",
        "SOUTH_C" to "South Central",
        "EAST" to "East",
        "NORTH_C" to "North Central",
        "COAST" to "Coast",
        "SOUTHERN" to "South",
        "WEST" to "West"
    )
    //Linked to Refresh data button
    LaunchedEffect(refreshData) {
        println("Refresh triggered")
        getData()
        //Cycle value to redraw graph
        displayVDBforecast.value = !displayVDBforecast.value
        displayVDBforecast.value = !displayVDBforecast.value
    }

    for (i in 0 until VDBforecast.size) {
        //Check for matching region and within timeFrame
        if (VDBlocation[i] in locationRename &&
            locationRename[VDBlocation[i]] == regionSelect.value &&
            VDBPoints.size <= timeFrame.value
        ) {
            //Add each point with the X being the index
            //SynthPoints.add(Point(i.toFloat(), Synthforecast[i].toFloat()))
            //Add corresponding date and time values
            if (filteredTime.size <= timeFrame.value) {
                filteredTime.add(timeValues[i])
                filteredDate.add(dateValues[i])
            }
        }
    }

    //Filter for VDB Forecast
    if (displayVDBforecast.value) {

        //Loop through the VDB forecast using filters
        for (i in 0 until VDBforecast.size) {
            //Check for matching region and within timeFrame
            if (VDBlocation[i] in locationRename &&
                locationRename[VDBlocation[i]] == regionSelect.value &&
                VDBPoints.size <= timeFrame.value
            ) {
                //Add each point with the X being the index
                VDBPoints.add(Point(i.toFloat(), VDBforecast[i].toFloat()))
                //Add corresponding date and time values if
                /*if (filteredTime.size <= timeFrame.value) {
                    filteredTime.add(timeValues[i])
                    filteredDate.add(dateValues[i])
                }*/
            }
        }
    }
    //Filter for ML Forecast
    if (displayMLforecast.value) {

        //Loop for filters
        for (i in 0 until MLforecast.size) {
            //Check for matching region and within timeFrame
            if (MLlocation[i] in locationRename &&
                locationRename[MLlocation[i]] == regionSelect.value &&
                MLPoints.size <= timeFrame.value
            ) {
                //Add each point with the X being the index
                MLPoints.add(Point(i.toFloat(), MLforecast[i].toFloat()))
                //Add corresponding date and time values
                /*if (filteredTime.size <= timeFrame.value) {
                    filteredTime.add(timeValues[i])
                    filteredDate.add(dateValues[i])
                }*/
            }
        }
    }
    //Filter for Synthesis Forecast
    if (displaySynthesis.value) {

        //Loop for filters
        for (i in 0 until Synthforecast.size) {
            //Check for matching region and within timeFrame
            if (Synthlocation[i] in locationRename &&
                locationRename[Synthlocation[i]] == regionSelect.value &&
                SynthPoints.size <= timeFrame.value
            ) {
                //Add each point with the X being the index
                SynthPoints.add(Point(i.toFloat(), Synthforecast[i].toFloat()))
                //Add corresponding date and time values
                /*if (filteredTime.size <= timeFrame.value) {
                    filteredTime.add(timeValues[i])
                    filteredDate.add(dateValues[i])
                }*/
            }
        }
    }





        LazyColumn(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 12.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            item { Spacer(modifier = Modifier.width(10.dp)) }
            //Refresh Button
            item {
                Box(
                    modifier = Modifier
                        .padding(12.dp)
                        .shadow(
                            elevation = 10.dp,
                            shape = RoundedCornerShape(8.dp)
                        )
                        .background(Color.White)
                ) {
                    Row(
                        modifier = Modifier.padding(
                            top = 3.dp,
                            bottom = 3.dp,
                            start = 6.dp,
                            end = 6.dp
                        ),
                        horizontalArrangement = Arrangement.Center,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(text = "Refreshed ", color = Color.Gray)
                        Text(text = current, color = Color.Gray)
                        Icon(
                            imageVector = Icons.Default.Refresh,
                            tint = Color.Gray,
                            contentDescription = "Favorite Icon",
                            modifier = Modifier
                                .size(42.dp)
                                .clickable {
                                    // Trigger variable to update
                                    SQL_error_message.value = ""
                                    //toggle variable
                                    refreshData = !refreshData
                                    println("Refresh button triggered")
                                }
                                .padding(8.dp)
                        )
                    }
                }
            }
            item {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 22.dp, vertical = 12.dp)
                        .shadow(
                            elevation = 10.dp,
                            shape = RoundedCornerShape(8.dp)
                        )
                        .background(Color.White)


                        .background(color = Color.White),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        modifier = Modifier,
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Spacer(modifier = Modifier.height(10.dp))
                        Text("MegaWatt Hours vs. Time")
                        Spacer(modifier = Modifier.height(10.dp))

                        Box(modifier = Modifier.fillMaxWidth()) {
                            MultiLineChart(
                                VDBPoints,
                                MLPoints,
                                SynthPoints,
                                filteredTime,
                                filteredDate,
                                timeFrame,
                                displayVDBforecast.value,
                                displayMLforecast.value,
                                displaySynthesis.value
                            )
                            Box(modifier = Modifier.align(Alignment.TopEnd)) {
                                labelBox(
                                    displayVDBforecast,
                                    displayMLforecast,
                                    displaySynthesis,
                                    filteredTime
                                )
                            }
                        }



                    }
                }
            }
            //User input
            item {
                userInput(
                    regionSelect, timeFrame,
                    displayVDBforecast,
                    displayMLforecast,
                    displaySynthesis
                )
            }
            //output error message if applicable
            item { Text(text = SQL_error_message.value, color = Color.Red) }
            item { Spacer(modifier = Modifier.height(55.dp)) }
        }




}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun Archive() {
    //Format for refresh button time
    val formatter = DateTimeFormatter.ofPattern("MM/dd hh:mm a")
    val current = LocalDateTime.now().format(formatter)
    var showPicker by remember { mutableStateOf(false) }
    val state = rememberDateRangePickerState()
    val startDate = remember { mutableStateOf("MM/DD/YYYY")}
    val endDate = remember { mutableStateOf("MM/DD/YYYY")}


    LazyColumn(
        modifier = Modifier
            .fillMaxWidth()
            .padding(top = 12.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        item { Spacer(modifier = Modifier.width(10.dp)) }
        //Refresh Button
        item {
            Box(
                modifier = Modifier
                    .padding(12.dp)
                    .shadow(
                        elevation = 10.dp,
                        shape = RoundedCornerShape(8.dp)
                    )
                    .background(Color.White)
            ) {
                Row(
                    modifier = Modifier.padding(
                        top = 3.dp,
                        bottom = 3.dp,
                        start = 6.dp,
                        end = 6.dp
                    ),
                    horizontalArrangement = Arrangement.Center,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(text = "Refreshed ", color = Color.Gray)
                    Text(text = current, color = Color.Gray)
                }
            }
        }
        item {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 22.dp, vertical = 12.dp)
                    .shadow(
                        elevation = 10.dp,
                        shape = RoundedCornerShape(8.dp)
                    )
                    .background(Color.White)


                    .background(color = Color.White),
                contentAlignment = Alignment.Center
            ) {
                Column(
                    modifier = Modifier,
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Spacer(modifier = Modifier.height(10.dp))
                    Text("MegaWatt Hours vs. Time")
                    Spacer(modifier = Modifier.height(10.dp))

                    Box(modifier = Modifier.fillMaxWidth()) {
                        //Display LineChart here
                        Text("Test Chart Here")
                        Box(modifier = Modifier.align(Alignment.TopEnd)) {
                            //Label Box if desired

                        }
                    }


                }
            }
        }
        //User input
        item {
            //Create New User Input
            archiveInput(startDate, endDate)

        }
        //output error message if applicable
        item { Text(text = "Create Error Message", color = Color.Red) }
        item { Spacer(modifier = Modifier.height(55.dp)) }
    }

}
@SuppressLint("MissingPermission")
//@Preview(widthDp = 350, heightDp = 800, backgroundColor = 1)
@Composable
fun Settings() {
    val context = LocalContext.current
    val formatter = DateTimeFormatter.ofPattern("MM/dd hh:mm a")
    val current = LocalDateTime.now().format(formatter)
    var expanded by remember { mutableStateOf(false) }
    val items = listOf("North", "North Central", "East",
        "West", "Far West", "South Central", "Coast", "South")
    var phoneError by remember {mutableStateOf("")}

    LazyColumn (
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        ) {
        item{Spacer(modifier = Modifier.height(20.dp))}
        item{Box(
            modifier = Modifier
                .padding(12.dp)
                .shadow(
                    elevation = 10.dp,
                    shape = RoundedCornerShape(8.dp)
                )
                .background(Color.White),
        ) {
            Row (
                modifier = Modifier.padding(top = 3.dp, bottom = 3.dp, start = 6.dp, end = 6.dp),
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(text = "Saved ", color = Color.Gray, fontSize = 12.sp)
                Text(text = current, color = Color.Gray, fontSize = 12.sp)
            }
        }
        }
        item {
            Box(
                modifier = Modifier
                    .width((GetScreenWidth() - 60).dp)
                    .padding(horizontal = 22.dp, vertical = 12.dp)
                    .shadow(
                        elevation = 10.dp,
                        shape = RoundedCornerShape(8.dp)
                    )
                    .background(Color.White),
                contentAlignment = Alignment.Center
            ) {
                Column(
                    modifier = Modifier.padding(0.dp)
                ) {
                    Text(
                        text = "Settings",
                        modifier = Modifier.padding(12.dp),
                        fontSize = 22.sp,
                        color = Color.Gray
                    )
                    Divider(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 12.dp),
                        thickness = 1.dp, color = Color.LightGray
                    )
                    //Row for Phone Slider
                    Row(
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "Phone Notifications",
                            modifier = Modifier.weight(1f),
                            color = Color.Gray,
                        )
                        Switch(
                            checked = phoneNotification.value,
                            onCheckedChange = { newValue -> phoneNotification.value = newValue },
                            modifier = Modifier.size(30.dp)
                        )
                        Spacer(modifier = Modifier.width(12.dp))
                    }
                    //Row for SMS Slider
                    Row(
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "SMS Notifications",
                            modifier = Modifier.weight(1f),
                            color = Color.Gray,
                        )
                        Switch(
                            checked = smsNotification.value,
                            onCheckedChange = { newValue -> smsNotification.value = newValue },
                            modifier = Modifier.size(30.dp),
                            //Only can be turned on if phone number is present
                            enabled = hasPhoneNumber(context)
                        )
                        Spacer(modifier = Modifier.width(12.dp))
                    }
                    Divider(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 12.dp),
                        thickness = 1.dp, color = Color.LightGray
                    )
                    Spacer(modifier = Modifier.height(12.dp))

                    //Phone Number input
                    TextField(
                        value = phoneNumber.value,
                        onValueChange = { phoneNumber.value = it },
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 12.dp, vertical = 6.dp),
                        label = {
                            Text(
                                text = "Phone Number",
                                color = Color.Gray,
                                fontSize = 12.sp
                            )
                        },
                        enabled = smsNotification.value && hasPhoneNumber(context),
                        keyboardOptions = KeyboardOptions(
                            keyboardType = KeyboardType.Phone,
                            imeAction = ImeAction.Done,
                            autoCorrect = false
                        )
                    )
                    //Power Input
                    TextField(
                        value = powerThreshold.value,
                        onValueChange = { powerThreshold.value = it },
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 12.dp, vertical = 6.dp),
                        label = {
                            Text(
                                text = "Power Threshold (MW)",
                                color = Color.Gray,
                                fontSize = 12.sp
                            )
                        },
                        enabled = phoneNotification.value || smsNotification.value,
                        keyboardOptions = KeyboardOptions(
                            keyboardType = KeyboardType.Number,
                            imeAction = ImeAction.Done,
                            autoCorrect = false
                        )
                    )
                    //Dropdown menu for region select
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 12.dp, vertical = 6.dp)
                            .border(2.dp, Color.LightGray)
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            TextButton(
                                onClick = { expanded = true },
                                modifier = Modifier.weight(1f)
                            ) {
                                Text(text = "Selected Region: ${regionValue.value}")
                            }
                            Icon(
                                imageVector = Icons.Default.ArrowDropDown,
                                contentDescription = null,
                                modifier = Modifier.clickable { expanded = true }
                            )
                        }
                        DropdownMenu(
                            //expand based on variable and if notifications turned on
                            expanded = expanded && (phoneNotification.value || smsNotification.value),
                            onDismissRequest = { expanded = false },
                            modifier = Modifier.align(Alignment.CenterEnd)
                        ) {
                            items.forEach { item ->
                                DropdownMenuItem(
                                    onClick = {
                                        regionValue.value = item
                                        expanded = false
                                    }
                                ) {
                                    Text(text = item)
                                }
                            }
                        }
                    }
                    Divider(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(start = 12.dp, end = 12.dp, top = 12.dp),
                        thickness = 1.dp, color = Color.LightGray
                    )

                    //Test Button
                    Row(
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "Test notifications",
                            modifier = Modifier.weight(1f),
                            color = Color.Gray,
                        )
                        Icon(
                            imageVector = Icons.Default.NotificationImportant,
                            tint = Color.Gray,
                            contentDescription = "Favorite Icon",
                            modifier = Modifier
                                .size(38.dp)
                                .clickable {
                                    //checks formatting
                                    phoneError = ""
                                    val phoneFormat = phoneNumber.value.matches(Regex("[0-9]+"))
                                    val powerFormat = powerThreshold.value.matches(Regex("[0-9]+"))
                                    if (!powerFormat && (smsNotification.value || phoneNotification.value)) {
                                        phoneError += "Incorrect power format \n"
                                    }
                                    if (!phoneFormat && smsNotification.value) {
                                        phoneError += "Incorrect phone number format \n"
                                    }

                                    // Trigger notification
                                    if (phoneNotification.value && powerFormat) {
                                        println("Phone Notification Triggered")
                                        //call phoneNotification function
                                        val message =
                                            "Notifcation test for power threshold ${powerThreshold.value} MW in the ${regionValue.value} region."
                                        showNotification(context, message)
                                    }
                                    if (smsNotification.value && powerFormat && phoneFormat) {
                                        println("SMS Notification Triggered")
                                        val message =
                                            "Notifcation test for power threshold ${powerThreshold.value} MW in the ${regionValue.value} region."
                                        sendText(phoneNumber.value, message)
                                    }
                                }
                                .padding(8.dp)
                        )
                        Spacer(modifier = Modifier.width(12.dp))
                    }

                    Divider(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 12.dp),
                        thickness = 1.dp, color = Color.LightGray
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                }
            }
        }
        item{Spacer(modifier = Modifier.height(10.dp))}
        item{Text(text = phoneError, modifier = Modifier.padding(4.dp), color = Color.Red)}
        item{Spacer(modifier = Modifier.height(40.dp))}
    }
}

@Composable
fun BottomNavigationBar(navController: NavHostController) {
    BottomNavigation(
        //backgroundColor = MaterialTheme.colorScheme.primary,
        backgroundColor = Color.LightGray,
        modifier = Modifier.padding(top = 12.dp)
    ) {
        val navBackStackEntry by navController.currentBackStackEntryAsState()
        val currentRoute = navBackStackEntry?.destination?.route
        val items = listOf(
            "Home" to Icons.Default.Home,
            "Forecast" to Icons.Default.AutoGraph,
            //"Archive" to Icons.Default.Leaderboard,
            "settings" to Icons.Default.Settings
        )
        items.forEach { (route, icon) ->
            BottomNavigationItem(
                icon = { Icon(icon, contentDescription = null,modifier = Modifier.size(32.dp)) },
                label = { Text(text = route) },
                selected = currentRoute == route,
                onClick = {
                    navController.navigate(route) {
                        // Pop up to the start destination of the graph to avoid building up a large stack of destinations
                        popUpTo(navController.graph.startDestinationId) {
                            saveState = true
                        }
                        // Avoid multiple copies of the same destination when re-selecting the same item
                        launchSingleTop = true
                        // Restore state when re-selecting a previously selected item
                        restoreState = true
                    }
                }
            )
        }
    }
}

suspend fun getData() {
    withContext(Dispatchers.IO) {
        //Empty arrays
        println("getData start")
        dateValues.clear()
        timeValues.clear()
        VDBforecast.clear()
        VDBlocation.clear()
        MLforecast.clear()
        MLlocation.clear()
        Synthforecast.clear()
        Synthlocation.clear()

        val connectionURL =
            "jdbc:jtds:sqlserver://tecafs.database.windows.net:1433;DatabaseName=TecafsSqlDatabase;" +
                    "user=tecafs2023@tecafs;" +
                    "password=Capstone50;" +
                    "encrypt=true;" +
                    "trustServerCertificate=false;" +
                    "hostNameInCertificate=*.database.windows.net;" +
                    "loginTimeout=30;" +
                    "ssl=require"
        var connection: Connection? = null // Declare connection variable as nullable
        try {
            Class.forName("net.sourceforge.jtds.jdbc.Driver")
            connection = DriverManager.getConnection(connectionURL)
            if (connection != null) {
                println("Connected to the database")
            } else {
                println("Failed to connect to the database")
                SQL_error_message.value += "Failed to connect to the database"
            }
            val statement = connection.createStatement()
            var resultSet =
                statement.executeQuery("SELECT location, date, hour, [forecast power]" +
                        "FROM VDB_daily_forecast " +
                        "ORDER BY location,time")
            // Process the results
            while (resultSet.next()) {
                val locationValue = resultSet.getString("location") ?: ""
                var dateValue = resultSet.getString("date") ?: ""
                var timeValue = resultSet.getString("hour") ?: ""
                var powerValue = resultSet.getString("forecast power") ?: ""
                //Format Date
                if (dateValue[0] == '0') {
                    dateValue = dateValue.substring(1)
                }
                dateValue = dateValue.replace('-', '/')
                //Format Time
                timeValue = convertTo12HourTime(timeValue)
                //Format Power
                val df = DecimalFormat("#.0")
                powerValue = df.format(powerValue.toDouble())
                //Add to arrays
                dateValues.add(dateValue)
                timeValues.add(timeValue)
                VDBforecast.add(powerValue)
                VDBlocation.add(locationValue)
            }
            val statement2 = connection.createStatement()
            val resultSet2 = statement2.executeQuery("SELECT location, [forecast power] " +
                    "FROM ML_forecast " +
                    "ORDER BY location")
            // Process the results
            while (resultSet2.next()) {
                val locationValue = resultSet2.getString("location") ?: ""
                var powerValue = resultSet2.getString("forecast power") ?: ""
                //Format Power
                val df = DecimalFormat("#.0")
                powerValue = df.format(powerValue.toDouble())
                //Add to arrays
                MLforecast.add(powerValue)
                MLlocation.add(locationValue)
            }

            val statement3 = connection.createStatement()
            val resultSet3 = statement3.executeQuery("SELECT location, [forecast power] " +
                    "FROM synthesize_forecast " +
                    "ORDER BY location, time")
            // Process the results
            while (resultSet3.next()) {
                val locationValue = resultSet3.getString("location") ?: ""
                var powerValue = resultSet3.getString("forecast power") ?: ""
                //Format Power
                val df = DecimalFormat("#.0")
                powerValue = df.format(powerValue.toDouble())
                //Add to arrays
                Synthforecast.add(powerValue)
                Synthlocation.add(locationValue)
            }

        } catch (e: Exception) {
            println(e)
            println("DateValues Size: " + dateValues.size)
            println("timeValues Size: " + timeValues.size)
            println("VDBforecast Size: " + VDBforecast.size)
            println("VDBforecast Size: " + VDBlocation.size)

            SQL_error_message.value += "Unexpected error: confirm internet connection \n"
        } finally {
            //Close Connection
            connection?.close()
        }
        println("getData finish")
    }
}

fun convertTo12HourTime(time24: String): String {
    val time24Hour = LocalTime.parse(time24, DateTimeFormatter.ofPattern("H:mm"))
    val formatter = DateTimeFormatter.ofPattern("h a")
    return time24Hour.format(formatter)
}

@Composable
private fun MultiLineChart(
    VDBPoints: MutableList<Point>,MLPoints: MutableList<Point>,SynthPoints: MutableList<Point>,
    filteredTime: MutableList<String>, filteredDate: MutableList<String> , timeFrame: MutableState<Int>,
    displayVDBforecast: Boolean, displayMLforecast: Boolean, displaySynforecast: Boolean) {
    if (VDBPoints.size ==337) {
        VDBPoints.removeAt(336)
    }
    println("VDB Points Size: " + VDBPoints.size)
    println("ML Points Size: " + MLPoints.size)
    println("Synth Points Size: " + SynthPoints.size)


    //Error check for empty data pull or array mismatch
    if (timeValues.size != VDBforecast.size || timeValues.size == 0) {
        //SQL_error_message += "Error: No time Values"
        println("There was an error with size formatting")
        timeValues.add("Error: Time values do not match forecast values")
        return
    }

    var xSteps = 0
    val xAxisData: AxisData
    //Format X step based on timeFrame
    if (timeFrame.value ==24) {
        //Make xStep based on hours
        xSteps = timeFrame.value
        xAxisData = AxisData.Builder()
            .axisStepSize(28.dp)
            .steps(xSteps)
            .axisLabelAngle(90f)
            .labelData { i -> if (i==0) {""} else {filteredTime[i-1]}   }
            .labelAndAxisLinePadding(10.dp)
            .axisLabelColor(Color.Gray)
            .axisLineColor(Color.DarkGray)
            .axisLabelFontSize(12.sp)
            .build()
    } else {
        //Otherwise xstep based on days
        xSteps = timeFrame.value / 24
        xAxisData = AxisData.Builder()
            .axisStepSize(if (timeFrame.value==72 ){14.dp}
                        else if (timeFrame.value==168) {6.dp}
                        else {4.dp} )
            .steps(xSteps)
            .axisLabelAngle(0f)
            .labelData { i -> if  (i <filteredDate.size && i!=0) {filteredDate[i-1]} else {""} }
            .axisOffset(20.dp)
            .labelAndAxisLinePadding(16.dp)
            .axisLabelColor(Color.Gray)
            .axisLineColor(Color.DarkGray)
            .axisLabelFontSize(12.sp)
            .build()
    }
    val ySteps = 9
    val yAxisData = AxisData.Builder()
        .steps(ySteps)
        .labelData { i ->
            var minVDB: Float = Float.POSITIVE_INFINITY
            var minML: Float = Float.POSITIVE_INFINITY
            var minSyn: Float = Float.POSITIVE_INFINITY
            var maxVDB: Float = Float.NEGATIVE_INFINITY
            var maxML: Float = Float.NEGATIVE_INFINITY
            var maxSyn: Float = Float.NEGATIVE_INFINITY
            if (VDBPoints.size>1) {
                minVDB =VDBPoints.minOf { it.y }
                maxVDB =VDBPoints.maxOf { it.y }
            }
            if (MLPoints.size>1) {
                minML =MLPoints.minOf { it.y }
                maxML =MLPoints.maxOf { it.y }
            }
            if (SynthPoints.size>1) {
                minSyn =SynthPoints.minOf { it.y }
                maxSyn =SynthPoints.maxOf { it.y }
            }
            val yMin = minOf(minVDB, minML,minSyn)
            val yMax = maxOf(maxVDB,maxML,maxSyn)
            val yScale = (yMax - yMin) / ySteps
            val yValue = ((i * yScale) + yMin)
            val formattedNumber = String.format("%.0f", floor(yValue.toDouble() / 100) * 100)
            formattedNumber
        }
        .labelAndAxisLinePadding(24.dp)
        .axisLabelColor(Color.Gray)
        .axisLineColor(Color.DarkGray)
        .build()

    val lines = mutableListOf<Line>()
    //Display if boolean true and points has data
    if (displayVDBforecast && VDBPoints.size >1) {
        lines.add(Line(
            dataPoints = VDBPoints,
            lineStyle = LineStyle(lineType = LineType.Straight(), color = TealBlue, width = 5.0F),
            selectionHighlightPopUp = SelectionHighlightPopUp(popUpLabel = { x, y ->
                indexLabel.value = x.toInt()
                ""
            },
                paddingBetweenPopUpAndPoint = 12.dp,
                labelColor = TealBlue,
                backgroundAlpha = 0.8F
            ),
            selectionHighlightPoint = SelectionHighlightPoint(
                color = Color.DarkGray,
                radius = 4.dp
            ),
        ))
    }
    //Display if boolean is true and points has data
    if (displayMLforecast && MLPoints.size >1) {
        lines.add(Line(
            dataPoints = MLPoints,
            lineStyle = LineStyle(lineType = LineType.Straight(), color = LimeGreen, width = 5.0F),
            selectionHighlightPopUp = SelectionHighlightPopUp(popUpLabel = { x, y ->
                indexLabel.value = x.toInt()
                ""
            },
                paddingBetweenPopUpAndPoint = 12.dp,
                labelColor = LimeGreen,
                backgroundAlpha = 0.8F
            ),
            selectionHighlightPoint = SelectionHighlightPoint(
                color = Color.DarkGray,
                radius = 4.dp
            ),
        ))
    }
    //Display if boolean is true and points has data
    if (displaySynforecast && SynthPoints.size >1) {
        lines.add(Line(
            dataPoints = SynthPoints,
            lineStyle = LineStyle(lineType = LineType.Straight(), color = Purple, width = 5.0F),
            selectionHighlightPopUp = SelectionHighlightPopUp(popUpLabel = { x, y ->
                indexLabel.value = x.toInt()
                ""
            },
                paddingBetweenPopUpAndPoint = 12.dp,
                labelColor = Purple,
                backgroundAlpha = 0.8F
            ),
            selectionHighlightPoint = SelectionHighlightPoint(
                color = Color.DarkGray,
                radius = 4.dp
            ),
        ))
    }
    //Case where all variables are off
    if (VDBPoints.size <= 1 && MLPoints.size <=1 && SynthPoints.size <= 1) {
        Line (dataPoints = listOf(Point(0f,0f)))
    }

    val data = LineChartData(
        linePlotData = LinePlotData(lines = lines),
        xAxisData = xAxisData,
        yAxisData = yAxisData,
        bottomPadding = 50.dp
    )
    LineChart(
        modifier = Modifier
            .width((GetScreenWidth() - 36).dp)
            .height(320.dp)
            .background(color = Color.Blue),
        lineChartData = data
    )

    //Create labels
    if (displayVDBforecast && VDBPoints.size >1 && indexLabel.value < timeFrame.value) {
        VDBlabel.value = VDBPoints[indexLabel.value].y.toString()
        timelabel.value = filteredTime[indexLabel.value]
    }
    if (displayMLforecast && MLPoints.size >1 && indexLabel.value < timeFrame.value) {
        MLlabel.value = MLPoints[indexLabel.value].y.toString()
        timelabel.value = filteredTime[indexLabel.value]
    }
    if (displaySynforecast && SynthPoints.size >1 && indexLabel.value < timeFrame.value) {
        Synthlabel.value = SynthPoints[indexLabel.value].y.toString()
        timelabel.value = filteredTime[indexLabel.value]
    }
}

@Composable
fun GetScreenWidth(): Int {
    val context = LocalContext.current
    val configuration = LocalConfiguration.current
    val screenWidth = remember(configuration) {
        context.resources.displayMetrics.widthPixels / context.resources.displayMetrics.density
    }
    return screenWidth.toInt()
}
@Composable
fun GetScreenHeight(): Int {
    val context = LocalContext.current
    val configuration = LocalConfiguration.current
    val screenHeight = remember(configuration) {
        context.resources.displayMetrics.heightPixels / context.resources.displayMetrics.density
    }
    return screenHeight.toInt()
}


@Composable
fun userInput(regionSelect: MutableState<String>,
              timeFrame: MutableState<Int>,
              displayVDBforecast: MutableState<Boolean>,
              displayMLforecast: MutableState<Boolean>,
              displaySynforecast: MutableState<Boolean>) {
    var expanded by remember { mutableStateOf(false) }
    val items = listOf("North", "North Central", "East",
                        "West", "Far West", "South Central", "Coast", "South")

    //Padding Values
    val rowPaddingVertical = 2.dp
    val rowPaddingHorizontal = 2.dp
    val columnPaddingHorizontal = 8.dp
    val columnPaddingVertical = 2.dp
    val headerPadding = 8.dp
    val inputPaddingHorizontal = 14.dp
    val inputPaddingVertical = 10.dp
    //Font Sizes
    val headerFont = 20.sp
    val inputFont = 14.sp
    //Sizes
    val checkBoxSize = 10.dp

    Box (
        modifier = Modifier
            .padding(horizontal = 12.dp)
            .shadow(
                elevation = 10.dp,
                shape = RoundedCornerShape(8.dp)
            )
            .background(Color.White),
        contentAlignment = Alignment.Center
    ) {
        //Column of Algorithm/Timeframe with Region underneath
        Column (
            //modifier = Modifier,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            //Row of just Algorithm and timeframe
            Row (
                modifier = Modifier.padding(0.dp)
            ) {
                //Algorithm Column
                Column(
                    modifier = Modifier.padding(horizontal = columnPaddingHorizontal, vertical = columnPaddingVertical)
                ) {
                    Text(modifier = Modifier.padding(headerPadding), text="Algorithms",fontSize = headerFont)
                    Row(
                        modifier = Modifier.padding(horizontal = rowPaddingHorizontal, vertical = rowPaddingVertical),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.Start
                    ) {
                        Checkbox(
                            modifier = Modifier
                                .padding(
                                    horizontal = inputPaddingHorizontal,
                                    vertical = inputPaddingVertical
                                )
                                .size(checkBoxSize),
                            checked = displayVDBforecast.value,
                            onCheckedChange = { displayVDBforecast.value = it }
                        )
                        Text(modifier = Modifier.padding(2.dp),text = "Vector Database",color = Color.Gray, fontSize = inputFont)
                    }
                    Row(
                        modifier = Modifier.padding(horizontal = rowPaddingHorizontal, vertical = rowPaddingVertical),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.Start
                    ) {
                        Checkbox(
                            modifier = Modifier
                                .padding(
                                    horizontal = inputPaddingHorizontal,
                                    vertical = inputPaddingVertical
                                )
                                .size(checkBoxSize),
                            checked = displayMLforecast.value,
                            onCheckedChange = { displayMLforecast.value = it }
                        )
                        Text(text = "Machine Learning", color = Color.Gray, fontSize = inputFont)
                    }
                    Row(
                        modifier = Modifier.padding(horizontal = rowPaddingHorizontal, vertical = rowPaddingVertical),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.Start

                    ) {
                        Checkbox(
                            modifier = Modifier
                                .padding(
                                    horizontal = inputPaddingHorizontal,
                                    vertical = inputPaddingVertical
                                )
                                .size(checkBoxSize),
                            checked = displaySynforecast.value,
                            onCheckedChange = { displaySynforecast.value = it }
                        )
                        Text(text = "Synthesis",color = Color.Gray, fontSize = inputFont)
                    }
                }
                //Time Frame Column
                Column(
                    modifier = Modifier.padding(horizontal = columnPaddingHorizontal, vertical = columnPaddingVertical)
                ) {
                    Text(modifier = Modifier.padding(headerPadding),text = "Forecast Window",fontSize = headerFont)
                    listOf("1 Day" to 24, "3 Days" to 72, "1 Week" to 168, "2 Weeks" to 336).forEach { (label, value) ->
                        Row(
                            modifier = Modifier.padding(horizontal = rowPaddingHorizontal, vertical = rowPaddingVertical),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            RadioButton(
                                modifier = Modifier
                                    .padding(
                                        horizontal = inputPaddingHorizontal,
                                        vertical = inputPaddingVertical
                                    )
                                    .size(checkBoxSize),
                                selected = timeFrame.value == value,
                                onClick = { timeFrame.value = value }
                            )
                            Text(text = label,color = Color.Gray, fontSize = inputFont)
                        }
                    }
                }
            }
            //Region Drop down menu
            Column(
                modifier = Modifier.padding(8.dp)
            ) {
                Box(
                    modifier = Modifier
                        .width(260.dp)
                        .border(2.dp, Color.LightGray)
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        TextButton(
                            onClick = { expanded = true },
                            modifier = Modifier.weight(1f)
                        ) {
                            Text(text="Selected Region: ${regionSelect.value}", fontSize = inputFont)
                        }
                        Icon(
                            imageVector = Icons.Default.ArrowDropDown,
                            contentDescription = null,
                            modifier = Modifier.clickable { expanded = true }
                        )
                    }
                    DropdownMenu(
                        expanded = expanded,
                        onDismissRequest = { expanded = false },
                        modifier = Modifier.align(Alignment.CenterEnd)
                    ) {
                        items.forEach { item ->
                            DropdownMenuItem(
                                onClick = {
                                    regionSelect.value = item
                                    expanded = false
                                }
                            ) {
                                Text(text = item)
                            }
                        }
                    }
                }
            }
            Spacer(modifier = Modifier.height(6.dp))
        }
    }
}

//@Preview
@Composable
fun displayTest () {
    var regionSelect = remember { mutableStateOf<String>("Coast") }
    var timeFrame = remember { mutableIntStateOf(24) }
    var displayVDBforecast = remember { mutableStateOf<Boolean>(false) }
    var displayMLforecast = remember { mutableStateOf<Boolean>(false) }
    var displaySynforecast = remember { mutableStateOf<Boolean>(false) }


    userInput(
        regionSelect = regionSelect,
        timeFrame = timeFrame,
        displayVDBforecast = displayVDBforecast,
        displayMLforecast = displayMLforecast,
        displaySynforecast = displaySynforecast
    )
}

fun sendText (phoneNumber: String, message: String) {
    val smsManager = SmsManager.getDefault()
    smsManager.sendTextMessage(
        ("+1$phoneNumber"),
        null,
        message,
        null,
        null
    )
}

@SuppressLint("MissingPermission")
fun hasPhoneNumber(context: Context): Boolean {
    val telephonyManager = getSystemService(context, TelephonyManager::class.java)
    val phoneNumber = telephonyManager?.line1Number
    return phoneNumber?.isNotEmpty() ?: false
}

@SuppressLint("MissingPermission")
fun showNotification(context: Context, message: String) {
    val channelId = "default_channel_id"
    val notificationId = 123

    // Create a NotificationManagerCompat instance
    val notificationManager = NotificationManagerCompat.from(context)

    // Create a notification channel for Android O and above
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
        val channel = NotificationChannel(
            channelId,
            "Default Channel",
            NotificationManager.IMPORTANCE_DEFAULT
        )
        notificationManager.createNotificationChannel(channel)
    }
    // Build the notification
    val builder = NotificationCompat.Builder(context, channelId)
        .setSmallIcon(android.R.drawable.ic_dialog_info)
        .setContentTitle("Power Forecast Warning")
        .setContentText(message)
        .setPriority(NotificationCompat.PRIORITY_HIGH)
    // Show the notification
    notificationManager.notify(notificationId, builder.build())
}

suspend fun checkValueExistsAsync(powerThreshold: Float): Boolean = withContext(Dispatchers.IO) {
    //Reset data and time
    dateAndTime = ""
    var connection: Connection? = null
    var exists = false
    val connectionURL = "jdbc:jtds:sqlserver://tecafs.database.windows.net:1433;DatabaseName=TecafsSqlDatabase;" +
            "user=tecafs2023@tecafs;" +
            "password=Capstone50;" +
            "encrypt=true;" +
            "trustServerCertificate=false;" +
            "hostNameInCertificate=*.database.windows.net;" +
            "loginTimeout=30;" +
            "ssl=require"
    val locationRename = mapOf(
        "Far West" to "FAR_WEST",
        "North" to "NORTH",
        "South Central" to "SOUTH_C",
        "East" to "EAST",
        "North Central" to "NORTH_C",
        "Coast" to "COAST",
        "South" to "SOUTHERN",
        "West" to "WEST"
    )

    try {
        println( "Location value: " + locationRename[regionValue.value])
        // Connect to the database
        connection = DriverManager.getConnection(connectionURL)
        //"SELECT location, date, hour, [forecast power] FROM Milvus_forecast"
        // Execute the query to check if the value exists
        val queryInput =
            "Select location, date, hour, [forecast power] " +
            "FROM VDB_daily_forecast " +
            "WHERE [forecast power] > $powerThreshold" +
            "AND location = '${locationRename[regionValue.value]}' " +
            "ORDER BY location,time"
        val statement = connection.prepareStatement(queryInput)
        val resultSet = statement.executeQuery()
        //Start with first occurance
        resultSet.next()
        // Check if the count is greater than 0
        val count = resultSet.getString(4).toFloat()
        exists = count > 0
        //Collect date and time if exists
        var dateValue = resultSet.getString("date") ?: ""
        var timeValue = resultSet.getString("hour") ?: ""
        if (dateValue[0] == '0') {
            dateValue = dateValue.substring(1)
        }
        dateValue = dateValue.replace('-', '/')
        timeValue = convertTo12HourTime(timeValue)
        //Store in variable for message
        dateAndTime = "$dateValue $timeValue"

    } catch (e: SQLException) {
        e.printStackTrace()
    } finally {
        // Close the connection
        connection?.close()
    }
    exists
}

fun isDisplayingNotification(context: Context): Boolean {
    val notificationManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
    // Check if the app is running on Android Oreo or higher
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
        val activeNotifications = notificationManager.activeNotifications
        for (notification in activeNotifications) {
            if (notification.id == 123) {
                return true
            }
        }
    } else {
        // For versions lower than Android Oreo, cannot check specific notifications
        // You can only check if there are active notifications in general
        return notificationManager.activeNotifications.isNotEmpty()
    }
    return false
}

@Composable
fun labelBox(displayVDBforecast: MutableState<Boolean>,
             displayMLforecast: MutableState<Boolean>,
             displaySynforecast: MutableState<Boolean>,
             filteredTime: MutableList<String>) {
    val labelFontSize = 12.sp
    val labelColor = Color.Gray

    Box(
        modifier = Modifier
            .padding(12.dp)
            .shadow(
                elevation = 2.dp,
                shape = RoundedCornerShape(8.dp)
            )
            .background(Color.White)
    ) {
        Column (
            modifier = Modifier.padding(4.dp)
        ) {
            if (displayVDBforecast.value) {
                Text(text = "VDB: " + VDBlabel.value + " MW",
                    fontSize = labelFontSize,
                    color = TealBlue)
            }
            if (displayMLforecast.value) {
                Text(text = "ML: " + MLlabel.value + " MW",
                    fontSize = labelFontSize,
                    color = LimeGreen)
            }
            if (displaySynforecast.value) {
                Text(text = "Synth: " + Synthlabel.value + " MW",
                    fontSize = labelFontSize,
                    color = Purple)
            }
            if (displayMLforecast.value || displayVDBforecast.value || displaySynforecast.value) {
                Text(text = "Time: " + timelabel.value,
                    fontSize = labelFontSize,
                    color = labelColor)
            }
        }
    }
}

@Composable
fun DateInput(label: String, date: MutableState<String>) {
    var text by remember { mutableStateOf(TextFieldValue(date.value)) }
    val focusManager = LocalFocusManager.current

    TextField(
        value = text,
        onValueChange = {
            if (it.text.length <= 10) { // limit the input to 8 characters (MM/DD/YYYY)
                text = it
                date.value = text.text
            }
        },
        label = { Text(label) },
        singleLine = true,
        keyboardOptions = KeyboardOptions(
            keyboardType = KeyboardType.Number,
            imeAction = ImeAction.Next
        ),
        keyboardActions = KeyboardActions(
            onNext = {
                focusManager.moveFocus(FocusDirection.Down)
            }
        )

    )
}

@Composable
fun archiveInput(
    startDate: MutableState<String>,
    endDate: MutableState<String>
) {
    Column {
        DateInput("Start Date", startDate)
        Spacer(modifier = Modifier.height(16.dp))
        DateInput("End Date", endDate)
    }
}


