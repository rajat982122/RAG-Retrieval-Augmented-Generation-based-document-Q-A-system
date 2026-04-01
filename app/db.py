import time

from app import config

try:
    import mysql.connector
except ImportError:
    mysql = None


def get_connection(use_database=True, retries=20, delay_seconds=3):
    if mysql is None:
        raise RuntimeError("mysql-connector-python is not installed.")

    # not sure if this is the best way but it works for Docker startup timing
    last_error = None

    for attempt in range(retries):
        possible_hosts = [
            (config.MYSQL_HOST, config.MYSQL_PORT),
            ("localhost", 3307),
            ("localhost", 3306),
        ]

        for host_name, host_port in possible_hosts:
            try:
                connection_settings = {
                    "host": host_name,
                    "port": host_port,
                    "user": config.MYSQL_USER,
                    "password": config.MYSQL_PASSWORD,
                }

                # The first connection creates the database if it does not exist yet.
                if use_database:
                    connection_settings["database"] = config.MYSQL_DATABASE

                connection = mysql.connector.connect(**connection_settings)
                return connection
            except mysql.connector.Error as error:
                last_error = error
                continue

        if attempt == retries - 1:
            raise RuntimeError(f"Could not connect to MySQL: {last_error}")
        time.sleep(delay_seconds)


def init_db():
    if mysql is None:
        print("MySQL connector not installed, so DB logging is disabled for now.")
        return

    # First connect without choosing a database, so we can create it.
    try:
        connection = get_connection(use_database=False)
        cursor = connection.cursor()

        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS {config.MYSQL_DATABASE}"
        )
        connection.commit()
        cursor.close()
        connection.close()

        # Now connect to the actual database and create the logs table.
        connection = get_connection(use_database=True)
        cursor = connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS query_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                question TEXT NOT NULL,
                answer LONGTEXT NOT NULL,
                response_time_ms INT NOT NULL
            )
            """
        )
        connection.commit()
        cursor.close()
        connection.close()
    except Exception as error:
        print(f"MySQL setup skipped for local run: {error}")


def log_query(question, answer, response_time_ms):
    if mysql is None:
        return

    try:
        connection = get_connection(use_database=True)
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO query_logs (question, answer, response_time_ms)
            VALUES (%s, %s, %s)
            """,
            (question, answer, response_time_ms),
        )

        connection.commit()
        cursor.close()
        connection.close()
    except Exception as error:
        # I did not want the whole API request to fail just because logging failed.
        print(f"MySQL logging failed: {error}")
