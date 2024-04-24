package com.example.mfs_tec

import android.annotation.SuppressLint
import android.os.Bundle
import android.os.StrictMode
import android.os.StrictMode.ThreadPolicy
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.BottomNavigation
import androidx.compose.material.BottomNavigationItem
import androidx.compose.material.Icon
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.DataUsage
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.example.mfs_tec.ui.theme.MFSTECTheme
import java.sql.Connection
import java.sql.DriverManager
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

class MainActivity : ComponentActivity() {
    @SuppressLint("UnusedMaterial3ScaffoldPaddingParameter")
    @OptIn(ExperimentalMaterial3Api::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            val navController = rememberNavController()
            
            MFSTECTheme {
                // A surface container using the 'background' color from the theme
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    Scaffold(
                        bottomBar = {
                            BottomNavigationBar(navController = navController)
                        }
                    ) {
                        NavHost(navController, startDestination = "Forecast") {
                            composable("Home") { Home() }
                            composable("Forecast") { Forecast() }
                            composable("Archive") { Archive() }
                            composable("settings") { Settings() }
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun Home() {
    val imagePainter = painterResource(id = R.drawable.ercot_map)

    Image(
        painter = imagePainter,
        contentDescription = "ERCOT Region Map",
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    )

}

@Preview(widthDp = 350, heightDp = 1000)
@Composable
fun Forecast() {
    val formatter = DateTimeFormatter.ofPattern("MM/dd HH:mm")
    val current = LocalDateTime.now().format(formatter)
    val refresh = remember { mutableStateOf(false) }

    LazyColumn(
        modifier = Modifier
            .fillMaxWidth()
            .padding(top = 12.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,

    ) {
        item {Spacer(modifier = Modifier.width(10.dp))}
        item {
            Box(
                modifier = Modifier
                    .padding(16.dp)
                    .background(color = Color.White)
                    .clip(shape = RoundedCornerShape(14.dp))
                    .shadow(
                        elevation = 4.dp,
                        clip = true
                    )
            ) {
                Row (
                    modifier = Modifier.padding(top = 3.dp, bottom = 3.dp, start = 6.dp, end = 6.dp),
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
                                // Update the state when the icon is clicked
                                refresh.value = !refresh.value
                            }
                            .padding(8.dp)
                    )
                }
            }
        }

        item{
            Box (
                modifier = Modifier
                    .padding(16.dp)
                    .background(color = Color.White)
                    .clip(shape = RoundedCornerShape(14.dp))
                    .shadow(
                        elevation = 4.dp,
                        clip = true
                    )
                    .size(width = 100.dp, height = 100.dp)
            ) {
                Text("test")
            }
        }





    }
}

@Composable
fun Archive() {
    Text(getData())

}

@Composable
fun Settings() {
    Text("Settings Screen")
}

@Composable
fun BottomNavigationBar(navController: NavHostController) {
    BottomNavigation(
        //backgroundColor = MaterialTheme.colorScheme.primary,
        backgroundColor = Color.Gray,

        modifier = Modifier.padding(top = 12.dp)
    ) {
        val navBackStackEntry by navController.currentBackStackEntryAsState()
        val currentRoute = navBackStackEntry?.destination?.route

        val items = listOf(
            "Home" to Icons.Default.Home,
            "Forecast" to Icons.Default.DataUsage,
            "Archive" to Icons.Default.History,
            "settings" to Icons.Default.Settings
        )

        items.forEach { (route, icon) ->
            BottomNavigationItem(
                icon = { Icon(icon, contentDescription = null,modifier = Modifier.size(32.dp)) },
                label = { Text(text = route) },
                selected = currentRoute == route,
                onClick = {
                    navController.navigate(route) {
                        // Pop up to the start destination of the graph to
                        // avoid building up a large stack of destinations
                        popUpTo(navController.graph.startDestinationId) {
                            saveState = true
                        }
                        // Avoid multiple copies of the same destination when
                        // re-selecting the same item
                        launchSingleTop = true
                        // Restore state when re-selecting a previously selected item
                        restoreState = true
                    }
                }
            )
        }
    }
}

fun getData(): String {
    //DriverManager.registerDriver(com.microsoft.sqlserver.jdbc.SQLServerDriver())
    //Class.forName("com.microsoft.sqlserver.jdbc.SQLServerDriver")
    val policy = ThreadPolicy.Builder().permitAll().build()
    StrictMode.setThreadPolicy(policy)

    val connectionURL = "jdbc:jtds:sqlserver://tecafs.database.windows.net:1433;database=TecafsSqlDatabase;user=tecafs2023@tecafs;password=Capstone50;" +
            "encrypt=true;" +
            "trustServerCertificate=false;" +
            "hostNameInCertificate=*.database.windows.net;loginTimeout=30;" +
            //"integratedSecurity=true;" +
            "sslProtocol=TLSv1.2;"
    var connection: Connection? = null
    var output = ""

    try {
        Class.forName("net.sourceforge.jtds.jdbc.Driver")
        println("driver working")
        connection = DriverManager.getConnection(connectionURL)
        // Use the connection...
        output ="Connected to Azure SQL Database!"
        println(output)

    } catch (e: Exception) {
        println(e)
        output = "Error connecting to Azure SQL Database: ${e.message}"
    } finally {
        //Close Connection
        connection?.close()
    }
    return output
}
