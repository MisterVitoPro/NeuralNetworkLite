
plugins {
    kotlin("jvm") version "1.7.10"
}

group = "ai.nueral-network"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation("org.testng:testng:6.9.10")
}

tasks.test {
    useTestNG()
}

//tasks.withType<KotlinCompile> {
//    kotlinOptions.jvmTarget = "1.8"
//}