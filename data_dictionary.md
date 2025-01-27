---

**`flarm_raw`**: This table stores raw FLARM/OGN sensor data, capturing details like sensor location (latitude, longitude, altitude), timestamps (server, sensor, and plane times), raw message content (26-byte FLARMv6 messages), and signal metrics (frequency, SNR, CRC validation). It includes technical corrections (e.g., NTP clock error, frequency adjustments) and distinguishes between OGN/FLARM message types. Data is partitioned by `hour` for efficient querying.

---

**`flights_data4`**: Tracks flight trajectories and airport estimates, including ICAO24 identifiers, first/last seen timestamps, departure/arrival airport candidates, and a `track` array storing positional waypoints (time, latitude, longitude). Metrics like horizontal/vertical distances to estimated airports and candidate airport lists are included. Partitioned by `day`.

---

**`flights_data5`**: An extended version of `flights_data4`, adding precise takeoff/landing details (times, coordinates), confirmed departure/destination airports, and expanded flight phase metadata. Like its predecessor, it uses `day` partitioning and retains core flight tracking features.

---

**`identification_data4`**: Focuses on aircraft identification, linking ICAO24 codes to emitter categories, flight identity codes, and message metadata. Includes sensor arrays contributing to detection and raw message content. Partitioned by `hour`.

---

**`operational_status_data4`**: Captures detailed aircraft system statuses, including TCAS capabilities, GPS accuracy metrics (NAC, NIC), antenna configurations, and safety flags (e.g., low TX power, TCAS advisories). Technical fields like `systemdesignassurance` and `airplanelength/width` provide operational context. Partitioned by `hour`.

---

**`position_data4`**: Stores real-time positional data (latitude, longitude, altitude) with quality indicators like NIC/NAC codes and surveillance status. Includes groundspeed, heading, and surface detection flags. Sensor arrays track contributing receivers. Partitioned by `hour`.

---

**`rollcall_replies_data4`**: Records aircraft responses to interrogations (e.g., ATC radar), including flight status, altitude, identity codes, and utility messages. Fields like `interrogatorid` and `downlinkrequest` clarify the context of replies. Partitioned by `hour`.

---

**`state_vectors_data4`**: Provides snapshot state vectors for aircraft, including position (lat/lon), velocity, altitude (barometric/geometric), and status flags (onground, alert, SPI). Lists `serials` of sensors detecting the aircraft. Partitioned by `hour`.

---

**`velocity_data4`**: Details velocity components (NS/EW speed, vertical rate) and motion attributes like supersonic flags, intent changes, and heading. Includes navigation accuracy (NAC) and barometric/geometric altitude references. Partitioned by `hour`.

---

Each table is partitioned chronologically (`hour` or `day`) to optimize performance, with sensor/aircraft metadata, raw messages, and derived metrics tailored to specific aviation data use cases (e.g., tracking, identification, operational monitoring).

---

## Table: `flarm_raw`

| Column | Type | Extra | Comment |
| --- | --- | --- | --- |
| sensortype | varchar |  | Type of the sensor |
| sensorlatitude | double |  | Latitude of the sensor |
| sensorlongitude | double |  | Longitude of the sensor |
| sensoraltitude | integer |  | Altitude of the sensor in meters (int) |
| timeatserver | double |  | Time at the ingestion server |
| timeatsensor | double |  | Time at the sensor (not used yet) |
| timestamp | double |  | Timestamp |
| timeatplane | integer |  | Time at which the position was broadcasted by the plane |
| rawmessage | varchar |  | 26 bytes FLARMv6 Raw message |
| crc | varchar |  | CRC pertaining to the FLARMv6 Raw message |
| rawsoftmessage | varchar |  | Soft bits, 4 per real bit, of a FLARMv6 Raw message |
| sensorname | varchar |  | Name of the sensor as broadcasted by OGN |
| ntperror | real |  | Error of the sensor's clock compared to NTP |
| userfreqcorrection | real |  | Dongle ppm correction set by the user |
| autofreqcorrection | real |  | Additional dongle ppm correction set by automation |
| frequency | double |  | Exact frequency at which the message was received |
| channel | integer |  | Channel on which the message was received (Important for SDR-based sensors) |
| snrdetector | double |  | SNR at the detector |
| snrdemodulator | double |  | SNR at the demodulator |
| typeogn | boolean |  | Type of the message (True > OGN, False > FLARM) |
| crccorrect | boolean |  | Is the CRC of the message correct? |
| hour | integer |  | **(Partition key)** |

---

## Table: `flights_data4`

| Column | Type | Extra | Comment |
| --- | --- | --- | --- |
| icao24 | varchar |  |  |
| firstseen | integer |  |  |
| estdepartureairport | varchar |  |  |
| lastseen | integer |  |  |
| estarrivalairport | varchar |  |  |
| callsign | varchar |  |  |
| track | array(row(time integer, latitude double, longitude double, ...)) |  |  |
| estdepartureairporthorizdistance | integer |  |  |
| estdepartureairportvertdistance | integer |  |  |
| estarrivalairporthorizdistance | integer |  |  |
| estarrivalairportvertdistance | integer |  |  |
| departureairportcandidatescount | integer |  |  |
| arrivalairportcandidatescount | integer |  |  |
| otherdepartureairportcandidates | array(row(icao24 varchar, horizdistance integer, vertdistance integer, ...)) |  |  |
| otherarrivalairportcandidates | array(row(icao24 varchar, horizdistance integer, vertdistance integer, ...)) |  |  |
| day | integer | **(Partition key)** |  |

---

## Table: `flights_data5`

| Column | Type | Extra | Comment |
| --- | --- | --- | --- |
| icao24 | varchar |  |  |
| firstseen | integer |  |  |
| estdepartureairport | varchar |  |  |
| lastseen | integer |  |  |
| estarrivalairport | varchar |  |  |
| callsign | varchar |  |  |
| track | array(row(time integer, latitude double, longitude double, ...)) |  |  |
| estdepartureairporthorizdistance | integer |  |  |
| estdepartureairportvertdistance | integer |  |  |
| estarrivalairporthorizdistance | integer |  |  |
| estarrivalairportvertdistance | integer |  |  |
| departureairportcandidatescount | integer |  |  |
| arrivalairportcandidatescount | integer |  |  |
| otherdepartureairportcandidates | array(row(icao24 varchar, horizdistance integer, vertdistance integer, ...)) |  |  |
| otherarrivalairportcandidates | array(row(icao24 varchar, horizdistance integer, vertdistance integer, ...)) |  |  |
| airportofdeparture | varchar |  |  |
| airportofdestination | varchar |  |  |
| takeofftime | integer |  |  |
| takeofflatitude | double |  |  |
| takeofflongitude | double |  |  |
| landingtime | integer |  |  |
| landinglatitude | double |  |  |
| landinglongitude | double |  |  |
| day | integer | **(Partition key)** |  |

---

## Table: `identification_data4`

| Column | Type | Extra | Comment |
| --- | --- | --- | --- |
| sensors | array(row(serial integer, mintime double, maxtime double, ...)) |  |  |
| rawmsg | varchar |  |  |
| mintime | double |  |  |
| maxtime | double |  |  |
| msgcount | bigint |  |  |
| icao24 | varchar |  |  |
| emittercategory | smallint |  |  |
| ftc | smallint |  |  |
| identity | varchar |  |  |
| hour | integer | **(Partition key)** |  |

---

## Table: `operational_status_data4`

| Column | Type | Extra | Comment |
| --- | --- | --- | --- |
| sensors | array(row(serial integer, mintime double, maxtime double, ...)) |  |  |
| rawmsg | varchar |  |  |
| icao24 | varchar |  |  |
| mintime | double |  |  |
| maxtime | double |  |  |
| msgcount | bigint |  |  |
| subtypecode | tinyint |  |  |
| unknowncapcode | boolean |  |  |
| unknownopcode | boolean |  |  |
| hasoperationaltcas | smallint |  |  |
| has1090esin | boolean |  |  |
| supportsairreferencedvelocity | smallint |  |  |
| haslowtxpower | smallint |  |  |
| supportstargetstatereport | smallint |  |  |
| supportstargetchangereport | smallint |  |  |
| hasuatin | boolean |  |  |
| nacv | tinyint |  |  |
| nicsupplementc | smallint |  |  |
| hastcasresolutionadvisory | boolean |  |  |
| hasactiveidentswitch | boolean |  |  |
| usessingleantenna | boolean |  |  |
| systemdesignassurance | tinyint |  |  |
| gpsantennaoffset | tinyint |  |  |
| airplanelength | integer |  |  |
| airplanewidth | double |  |  |
| version | tinyint |  |  |
| nicsupplementa | boolean |  |  |
| positionnac | double |  |  |
| geometricverticalaccuracy | integer |  |  |
| sourceintegritylevel | tinyint |  |  |
| barometricaltitudeintegritycode | smallint |  |  |
| trackheadinginfo | smallint |  |  |
| horizontalreferencedirection | boolean |  |  |
| hour | integer | **(Partition key)** |  |

---

## Table: `position_data4`

| Column | Type | Extra | Comment |
| --- | --- | --- | --- |
| sensors | array(row(serial integer, mintime double, maxtime double, ...)) |  |  |
| rawmsg | varchar |  |  |
| mintime | double |  |  |
| maxtime | double |  |  |
| msgcount | bigint |  |  |
| icao24 | varchar |  |  |
| nicsuppla | boolean |  |  |
| hcr | double |  |  |
| nic | smallint |  |  |
| survstatus | smallint |  |  |
| nicsupplb | boolean |  |  |
| odd | boolean |  |  |
| baroalt | boolean |  |  |
| lat | double |  |  |
| lon | double |  |  |
| alt | double |  |  |
| nicsupplc | boolean |  |  |
| groundspeed | double |  |  |
| gsresolution | double |  |  |
| heading | double |  |  |
| timeflag | boolean |  |  |
| surface | boolean |  |  |
| hour | integer | **(Partition key)** |  |

---

## Table: `rollcall_replies_data4`

| Column | Type | Extra | Comment |
| --- | --- | --- | --- |
| sensors | array(row(serial integer, mintime double, maxtime double, ...)) |  |  |
| rawmsg | varchar |  |  |
| mintime | double |  |  |
| maxtime | double |  |  |
| msgcount | bigint |  |  |
| icao24 | varchar |  |  |
| message | varchar |  |  |
| isid | boolean |  |  |
| flightstatus | tinyint |  |  |
| downlinkrequest | tinyint |  |  |
| utilitymsg | tinyint |  |  |
| interrogatorid | tinyint |  |  |
| identifierdesignator | tinyint |  |  |
| valuecode | smallint |  |  |
| altitude | double |  |  |
| identity | varchar |  |  |
| hour | integer | **(Partition key)** |  |

---

## Table: `state_vectors_data4`

| Column | Type | Extra | Comment |
| --- | --- | --- | --- |
| time | integer |  |  |
| icao24 | varchar |  |  |
| lat | double |  |  |
| lon | double |  |  |
| velocity | double |  |  |
| heading | double |  |  |
| vertrate | double |  |  |
| callsign | varchar |  |  |
| onground | boolean |  |  |
| alert | boolean |  |  |
| spi | boolean |  |  |
| squawk | varchar |  |  |
| baroaltitude | double |  |  |
| geoaltitude | double |  |  |
| lastposupdate | double |  |  |
| lastcontact | double |  |  |
| serials | array(integer) |  |  |
| hour | integer | **(Partition key)** |  |

---

## Table: `velocity_data4`

| Column | Type | Extra | Comment |
| --- | --- | --- | --- |
| sensors | array(row(serial integer, mintime double, maxtime double, ...)) |  |  |
| rawmsg | varchar |  |  |
| mintime | double |  |  |
| maxtime | double |  |  |
| msgcount | bigint |  |  |
| icao24 | varchar |  |  |
| supersonic | boolean |  |  |
| intentchange | boolean |  |  |
| ifrcapability | boolean |  |  |
| nac | smallint |  |  |
| ewvelocity | double |  |  |
| nsvelocity | double |  |  |
| baro | boolean |  |  |
| vertrate | double |  |  |
| geominurbaro | double |  |  |
| heading | double |  |  |
| velocity | double |  |  |
| hour | integer | **(Partition key)** |  |

---

**Note:**

- **Partition key** denotes the column used for partitioning.
- “Extra” field is shown if there was an explicit mention (e.g., partition key). Otherwise, it is left blank.