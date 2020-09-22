# Want to add function to `vertica_functions` that can query a table, get distinct column_names,
# and use it to make JOINs easier/reduce errors when making edits to code.
# Obviously the JOINing would only work when tables have the same column name.
# Could fix some issues when writing the subqueries aliasing column names.
# Or could perform the join and then include custom column JOINs when we know when there will be an issue.
# Needs to work with pre-built tables and subqueries.


# coding: utf-8

# # Dependencies

# In[ ]:


import pandas as pd
import numpy as np
from IPython.display import Audio
import requests
from bs4 import BeautifulSoup
#import vertica_python as vp
import sched
import time
import inspect
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# # General

# In[ ]:

def clusters(cluster):
    """
    Accepts a cluster name and returns the host IP address
    
    Parameters
    ----------
    cluster: str
        cluster name
    """
    cluster = cluster.lower()
    cluster_dict = {
        'aardvark' : '44.128.65.245',
        'antelope' : 'vertica-antelope.us-zone1.healtheintent.net',
        'badger' : '44.128.65.246',
        'buffalo' : 'vertica-buffalo.us-zone1.healtheintent.net',
        'caribou' : 'vertica-caribou.us-zone1.healtheintent.net',
        'chinchilla' : '44.128.65.247',
        'deer' : 'vertica-deer.us-zone1.healtheintent.net',
        'dormouse' : '44.128.65.248',
        'echidna' : '44.128.65.24',
        'ferret' : '44.128.76.70',
        'gerbil' : '44.128.76.72',
        'hamster' : '44.128.76.231'
    }
    return cluster_dict[cluster]

def alarm():
    """
    Sounds an alarm for 5 seconds.
    """
    framerate = 44100
    t = np.linspace(0,5,framerate*5)
    data = np.sin(2*np.pi*220*t) + np.sin(2*np.pi*224*t)
    return Audio(data,rate=framerate, autoplay = True)

def timer(hours = 0, minutes = 0, seconds = 0, alarm = alarm()):
    """
    Sets a timer for user-defined time.
    
    Parameters
    ----------
    hours: int (default 0)
        number of hours until timer expires.
    minutes: int (default 0)
        number of minutes until timer expires.
    seconds: int (default 0)
        number of seconds until timer expires.
    alarm: object (default alarm())
        sounds an alarm when timer expires
    """
    # convert to seconds
    h = 60*60*hours
    m = 60*minutes
    s = seconds
    # add up seconds
    t = h + m + s
    # create scheduler
    scheduler = sched.scheduler(timefunc = time.time, delayfunc = time.sleep)
    scheduler.enter(delay = 0, priority = 0, action = time.sleep(t))
    if alarm:
        return alarm
    elif not alarm:
        return('Done')
    
def equal_in(some_list):
    """
    Checks length of some_list and returns either a string or tuple for filtering a query.
    
    Parameters
    ----------
    some_list: list
        a list of objects
    """
    
    try:
        # assert that some_list is actually a list
        assert type(some_list) == list
        # if list is length 1, use "="
        if len(some_list) == 1:
            return "= '{}'".format(some_list[0])
        # else use "IN"
        elif len(some_list) > 1:
            return "IN {}".format(tuple(some_list))
    # raise assertion error if some_list isn't a list
    except AssertionError as e:
        e.args = (f"{some_list} is not of type list.",)
        raise


# # Vertica

# In[ ]:


class vertica_functions:
    """
    A set of useful functions to be used with Vertica SQL.
    """
    def __init__(self, host, user, password, port = 5433, database = 'default'):
        self.host = host
        self.user = user
        self.password = password
        self.port = port
        self.database = database
    
    def query_df(self, query, limit = None ):
        """
        Takes a Vertica SQL query and returns a pd.DataFrame()
        
        Parameters
        ----------
        query: a single or multi-lined string
            a Vertica SQL query
        limit: int (optional)
            limit the number of rows the query returns
        """
        
        connection_info = {
            'host' : self.host,
            'user' : self.user,
            'password' : self.password,
            'port' : self.port,
            'database' : self.database
        }
        with vp.connect(**connection_info) as connection:
            cursor = connection.cursor(cursor_type = 'dict')
            if limit is not None:
                query += ' LIMIT {limit}'.format(**{
                    'limit' : limit
                })
            cursor.execute(operation = query)
            return pd.DataFrame(cursor.fetchall())
    
    def pivot(self, table, column, alias, prefix = '', schema = None, aggfunc = 'MAX', THEN = '1', ELSE = '0', include_null = False):
        """
        Takes a `pd.Series()` from a returned Vertica query and creates `.nunique()` CASE statements
        for each `.unique()` element in the `pd.Series()`.
        This only returns the CASE statements that can be inserted into the query.
        By default it excludes `None` values (NULLs in Vertica)
        
        Parameters
        ----------
        table: str
            the table where the column will come from
        column: str
            the column to pivot
        alias: str
            table alias to use
        prefix: str (default '')
            prefix to be added to CASE column names
        schema: str (default None)
            the schema where the table will come from
        aggfunc: str (default 'MAX')
            function to use on the CASE statements
        THEN: str (default '1')
            what to return if WHEN returns TRUE
        ELSE: str (default '0')
            what to return if WHEN returns FALSE
        include_null: bool (default False)
            whether or not to include NULL values. If `True`, `None` will be a returned CASE.
        """
#         get unique elements from column of interest
        tbl = '({table})'.format(**{
            'table' : table
        })
        distinct = '''
        SELECT DISTINCT
            {alias}.{column}
        FROM
        '''.format(**{
            'alias' : alias,
            'column' : column
        })
        if schema is None:
            distinct += f"{tbl} {alias}"
        else:
            distinct += f"{schema}.{table} {alias}"
        d = self.query_df(distinct)
        if include_null == False:
            d = d.dropna()
        # loop through the values and create the query
        case = ""
        for i in d[column]:
            case += f"""{aggfunc}(
            CASE
                WHEN {alias}.{column} = '{i}'
                    THEN {THEN}
                    ELSE {ELSE}
            END) AS '{prefix}{i}'"""
            if i != d[column].iloc[-1]:
                case += ',\n'
        return case
    
    def check_where(self, frame, alias = None):
        """
        Returns the WHERE statement for a query based on the current frames parameters.
        Does not include date logic!

        Parameters
        ----------
        frame: function
            the current frame
        alias: str (default None)
            optional alias when using a subquery
        """
        
        # get args and values from current frame
        args, _, _, values = inspect.getargvalues(frame)
        # create dictionary of args and their respective values (if there are any); exclude date
        o = {a : values[a] for a in args[1:] if values[a] and a != 'date'}
        # make list of dictionary keys
        n = list(o)
        # get number of keys
        ln = len(n)
        # begin empty WHERE clause
        q = ""
        # check length of keys
        if ln > 0:
            # write first WHERE statement
            # add alias if given
            q += f"""
            WHERE
                {['' if a is None else f'{a}.' for a in [alias]][0]}{n[0]} {equal_in(o[n[0]])}
            """
            # write remaining WHERE statements
            # add alias if given
            for k in n[1:]:
                q += f"""
                AND {['' if a is None else f'{a}.' for a in [alias]][0]}{k} {equal_in(o[k])}
                """
            # return WHERE statements
            return q
        else:
            # return empty string
            return q


# # Client

# In[ ]:

class client_functions(vertica_functions):
    """
    A set of userful functions to be used with client schemas.
    """
    
    def __init__(self, host, user, password, schema, population_id):
        super().__init__(host, user, password)
        self.schema = schema
        self.population_id = population_id
    
    def alias_query(self, alias = None):
        """
        Returns a query to search the ontology tables for specific aliases.
        
        Parameters
        ----------
        alias: list (default None)
            list of aliases to query
        """
        
        frame = inspect.currentframe()
        where = self.check_where(frame)
        q = f"""
        SELECT DISTINCT
            ont.population_id,
            ont.code_oid,
            ont.code_system_id,
            ontc.alias
        FROM
            {self.schema}.PH_D_Ontology ont
            JOIN {self.schema}.PH_D_Ontology_Concept_Alias ontc ON (
                ont.population_id = ontc.population_id
                AND ont.concept_id = ontc.concept_id
                AND ont.context_id = ontc.context_id
                AND ont.population_id = '{self.population_id}'
            )
        """
        q += where
        return q
    
    def code_oid_query(self, code_oids):
        """
        Returns a query to search the ontology tables for specific code_oids.
        
        Parameters
        ----------
        code_oid: list
            list of code_oids to query
        """
        
        code_oid = equal_in(code_oids)
        q = f"""
        SELECT DISTINCT
            ont.population_id,
            ont.code_oid,
            ont.code_system_id,
            ontc.alias
        FROM
            {self.schema}.PH_D_Ontology ont
            JOIN {self.schema}.PH_D_Ontology_Concept_Alias ontc ON (
                ont.population_id = ontc.population_id
                AND ont.concept_id = ontc.concept_id
                AND ont.context_id = ontc.context_id
                AND ont.population_id = '{self.population_id}'
                AND ont.code_oid {code_oid}
            )
        """
        return q
    
    def prsnl_group_query(self, prsnl_group_id = None, prsnl_group_mnemonic = None):
        """
        Return a query for filtering prsnl_ids by prsnl_group_id.
        
        Parameters
        ----------
        prsnl_group_id: list (default None)
            list of prsnl_group_ids to filter
        prsnl_group_mnemonic: list (default None)
            list of prsnl_group_mnemonics to filter
        """
        
        frame = inspect.currentframe()
        where = self.check_where(frame)
        g = f"""
        SELECT DISTINCT
            prsnl_group_id,
            prsnl_group_mnemonic,
            prsnl_group_name
        FROM
            {self.schema}.PH_D_Personnel_Group
        """
        g += where
        q = """
        SELECT DISTINCT
            g.*,
            p.prsnl_id
        FROM
            ({g}) g
            JOIN {schema}.PH_D_Personnel_Personnel_Group_Reltn p ON (
                g.prsnl_group_id = p.prsnl_group_id
            )
        """.format(**{
            'g' : g,
            'schema' : self.schema
        })
        return q
    
    def source_id_query(self, source_id, table = 'enc'):
        """
        Returns a query for searching Encounters/Procedures/PPRs by source_id.
        
        Parameters
        ----------
        source_id: list
            the source_id(s) to search for
        table: str (default 'enc')
            the table to search for the source_id (accepts 'enc', 'proc', 'ppr')
        """
        
        source = equal_in(source_id)
        if table == 'enc':
            q = f"""
            SELECT DISTINCT
                enc.empi_id,
                prov.prsnl_id,
                prov.provider_id,
                enc.source_id,
                enc.service_date_id,
                enc.service_dt_tm,
                enc.encounter_type_code,
                enc.encounter_type_coding_system_id
            FROM
                {self.schema}.PH_F_Encounter enc
                JOIN {self.schema}.PH_F_Encounter_Personnel_Reltn epr ON (
                    enc.population_id = epr.population_id
                    AND enc.empi_id = epr.empi_id
                    AND enc.encounter_id = epr.encounter_id
                    AND enc.source_id {source}
                    AND enc.population_id = '{self.population_id}'
                )
                JOIN {self.schema}.PH_D_Provider prov ON (
                    epr.provider_id = prov.provider_id
                )
            """
        elif table == 'proc':
            q = f"""
            SELECT DISTINCT
                proc.empi_id,
                prov.prsnl_id,
                prov.provider_id,
                proc.source_id,
                proc.service_start_date_id,
                proc.service_start_dt_tm,
                proc.procedure_code,
                proc.procedure_coding_system_id
            FROM
                {self.schema}.PH_F_Procedure proc
                JOIN {self.schema}.PH_F_Procedure_Personnel_Reltn ppr ON (
                    proc.population_id = ppr.population_id
                    AND proc.empi_id = ppr.empi_id
                    AND proc.procedure_id = ppr.procedure_id
                    AND proc.source_id {source}
                    AND proc.population_id = '{self.population_id}'
                )
                JOIN {self.schema}.PH_D_Provider prov ON (
                    ppr.provider_id = prov.provider_id
                )
            """
        elif table == 'ppr':
            q = f"""
            SELECT DISTINCT
                ppr.empi_id,
                prov.prsnl_id,
                prov.provider_id,
                ppr.source_id,
                ppr.begin_date_id,
                ppr.begin_dt_tm,
                ppr.end_date_id,
                ppr.end_dt_tm,
                ppr.relationship_type_code,
                ppr.relationship_type_coding_system_id
            FROM
                {self.schema}.PH_F_Person_Personnel_Reltn ppr
                JOIN {self.schema}.PH_D_Provider prov ON (
                    ppr.provider_id = prov.provider_id
                    AND ppr.source_id {source}
                    AND ppr.population_id = '{self.population_id}'
                )
            """
        else:
            raise ValueError(f"'{table}' is not a valid table. Must be one of 'enc', 'proc', or 'ppr'.")
        return q
    
    def person_query(self, empi_id = None, population_id = None):
        """
        Returns a query that searches for empi_id(s) and returns generic information (e.g. gender, birthdate, deceased, etc.)
        
        Parameters
        ----------
        empi_id: list (default None)
            empi_id(s) to search for
        population_id: list (default None)
            population_id to filter
        """
        
        frame = inspect.currentframe()
        where = self.check_where(frame)
        q = """
        SELECT DISTINCT
            population_id,
            empi_id,
            full_name,
            birth_date_id,
            birth_date,
            deceased,
            deceased_date_id,
            deceased_dt_tm,
            gender_primary_display
        FROM
            {schema}.PH_D_Person
        """.format(**{
            'schema' : self.schema
        })
        q += where
        return q
    
    def prsnl_alias_query(self, prsnl_id = None, prsnl_alias_id = None, prsnl_alias_type = None):
        """
        Returns a query from the PH_D_Personnel_Alias table.
        
        Parameters
        ----------
        prsnl_id: list (default None)
            for searching a prsnl_id
        prsnl_alias_id: list (default None)
            for searching a prsnl_alias_id
        prsnl_alias_type: list (default None)
            for searching a prsnl_alias_type (e.g. 'NPI')
        """
        
        frame = inspect.currentframe()
        where = self.check_where(frame)
        q = """
        SELECT DISTINCT
            prsnl_id,
            prsnl_alias_id,
            prsnl_alias_type
        FROM
            {schema}.PH_D_Personnel_Alias
        """.format(**{
            'schema' : self.schema
        })
        q += where
        return q
    
    def provider_query(self, provider_id = None, prsnl_id = None):
        """
        Returns a query for searching the PH_D_Provider table.
        
        Parameters
        ----------
        provider_id: list (default None)
            list of provider_ids to search
        prsnl_id: list (default None)
            list of prsnl_ids to search
        """
        
        frame = inspect.currentframe()
        where = self.check_where(frame)
        q = """
        SELECT DISTINCT
            provider_id,
            prsnl_id,
            name,
            given_name,
            family_name
        FROM
            {schema}.PH_D_Provider
        """.format(**{
            'schema' : self.schema
        })
        q += where
        return q
    
    def benefit_coverage_query(self, partition_id = None, empi_id = None, member_id = None, date = None, population_id = None):
        """
        Returns a query for the PH_F_Person_Benefit_Coverage table.
        
        Parameters
        ----------
        partition_id: list (default None)
            filter the partition_ids
        empi_id: list (default None)
            filter the empi_ids
        member_id: list (default None)
            filter the member_ids
        date: str (default None)
            filter the begin and end dates
        population_id: list (default None)
            filter the population_ids
        """
        
        # check frame for parameters
        frame = inspect.currentframe()
        # write where clause
        where = self.check_where(frame, alias='q')
        # write start query
        q = f"""
        SELECT DISTINCT
            SPLIT_PART(SPLIT_PART(pbc_id, ':', 2), '/', 1) AS partition_id,
            empi_id,
            member_id,
            benefit_type_primary_display,
            begin_dt_tm,
            begin_date_id,
            end_date_id,
            end_dt_tm,
            claim_uid,
            population_id
        FROM
            {self.schema}.PH_F_Person_Benefit_Coverage
        """
        # write subquery
        q = f"""
        SELECT DISTINCT
            q.*
        FROM
            ({q}) q
        """
        # add WHERE clause
        q += where
        # add date filter if given
        if date is not None:
            # check if WHERE clause was added
            if len(where) > 0:
                q += f"""
                AND (
                    q.begin_dt_tm IS NULL
                    OR q.begin_dt_tm <= '{date}'
                )
                AND (
                    q.end_dt_tm IS NULL
                    OR q.end_dt_tm >= '{date}'
                )
                """
            # if WHERE clause not added, start new one
            else:
                q += f"""
                WHERE
                    (
                        q.begin_dt_tm IS NULL
                        OR q.begin_dt_tm <= '{date}'
                    )
                    AND (
                        q.end_dt_tm IS NULL
                        OR q.end_dt_tm >= '{date}'
                    )
                """
        # return query
        return q
    
    def person_alias_query(self, empi_id = None, alias_type_code = None, alias = None, population_id = None):
        """
        Returns a query to search the PH_D_Person_Alias table.
        
        Parameters
        ----------
        empi_id: list
            list of empi_ids to query
        alias_type_code: list
            list of alias_type_codes to query
        alias: list
            list of alias(s) to query
        population_id: list (default None)
            population_id to filter
        """
        
        frame = inspect.currentframe()
        where = self.check_where(frame)
        q = f"""
        SELECT DISTINCT
            population_id,
            empi_id,
            alias_type_primary_display,
            alias_type_code,
            alias_type_coding_system_id,
            alias,
            source_description,
            source_type
        FROM
            {self.schema}.PH_D_Person_Alias
        """
        q += where
        return q
    
    def organization_query(self, org_id = None, org_name = None):
        """
        Returns a query with org_name, org_class_name, and the associated prsnl_id.
        
        Parameters
        ----------
        org_id: list (default None)
            list of org_ids to filter on
        org_name: list (default None)
            list of org_names to filter
        """
        
        frame = inspect.currentframe()
        where = self.check_where(frame)
        o = f"""
        SELECT DISTINCT
            org_id,
            org_name
        FROM
            {self.schema}.PH_D_Organization
        """
        o += where
        q = f"""
        SELECT DISTINCT
            o.org_id,
            o.org_name,
            p.prsnl_id
        FROM
            ({o}) o
            FULL OUTER JOIN {self.schema}.PH_D_Organization_Personnel_Reltn p ON (
                o.org_id = p.org_id
            )
        """
        return q
    
    def organization_class_query(self, org_class_id = None, org_class_name = None):
        """
        Returns a query with org_name, org_class_name, and the associated prsnl_id.
        
        Parameters
        ----------
        org_class_id: list (default None)
            list of org_class_ids to filter on
        org_class_name: list (default None)
            list of org_class_names to filter
        """
        
        frame = inspect.currentframe()
        where = self.check_where(frame)
        o = f"""
        SELECT DISTINCT
            org_class_id,
            org_class_name
        FROM
            {self.schema}.PH_D_Organization_Class
        """
        o += where
        q = f"""
        SELECT DISTINCT
            c.org_class_id,
            c.org_class_name,
            o.org_id
        FROM
            ({o}) c
            FULL OUTER JOIN {self.schema}.PH_D_Organization_Organization_Class_Reltn o ON (
                c.org_class_id = o.org_class_id
            )
        """
        return q

    def source_query(self, source_type = None, partition = None, table_name = None):
        """
        Returns a query for searching data partitions, source types, and source descriptions.
        
        Parameters
        ----------
        source_type: list
            list of source types to filter on
        partition: list
            list of data partitions to filter on
        table_name: list
            list of table names to filter on. Accepts PH_F_Person_Personnel_Reltn, PH_F_Encounter, or PH_F_Procedure.
        """
        
        frame = inspect.currentframe()
        where = self.check_where(frame)
        ppr_q = f"""
        SELECT DISTINCT
            'PH_F_Person_Personnel_Reltn' AS table_name,
            SPLIT_PART(SPLIT_PART(ppr_id, ':', 2), '/', 1) AS partition,
            source_description,
            source_type
        FROM
            {self.schema}.PH_F_Person_Personnel_Reltn
        """
        enc_q = f"""
        SELECT DISTINCT
            'PH_F_Encounter' AS table_name,
            SPLIT_PART(encounter_id, ':', 1) AS partition,
            source_description,
            source_type
        FROM
            {self.schema}.PH_F_Encounter
        """
        proc_q = f"""
        SELECT DISTINCT
            'PH_F_Procedure' AS table_name,
            SPLIT_PART(SPLIT_PART(procedure_id, ':', 2), '/', 1) AS partition,
            source_description,
            source_type
        FROM
            {self.schema}.PH_F_Procedure
        """
        q = f"""
        SELECT DISTINCT
            q.*
        FROM
            (({ppr_q})
            
            UNION
            
            ({enc_q})
            
            UNION
            
            ({proc_q})) q
        """
        q += where
        return q

# # Attribution

# In[ ]:


class attribution_functions(client_functions):
    """
    A set of useful functions to be used with generic attribution algorithms.
    """
    
    def __init__(self, host, user, password, schema, population_id, context_id = '8FBD43EF0885489AA9FF961D66294839'):
        super().__init__(host, user, password, schema, population_id)
        self.context_id = context_id
    
    def COC(v):
        """
        Calculate the Continuity of Care Index

        Parameters
        ----------
        v : 1D-array
            an array of the providers for each visit

        Returns
        -------
        float (2 decimals)
        """

        uv = np.unique(v)
        k = len(uv)
        N = len(v)

        s = 0
        for k in uv:
            s += np.sum(v == k)**2
        return np.round((s - N) / (N *(N - 1)), 2)
    
    def alias_query(self, alias = None):
        """
        Returns a query to search the ontology tables for specific aliases.
        
        Parameters
        ----------
        alias: list (default None)
            list of aliases to query
        """
        
        frame = inspect.currentframe()
        where = self.check_where(frame)
        q = f"""
        SELECT DISTINCT
            ont.population_id,
            ont.code_oid,
            ont.code_system_id,
            ontc.alias
        FROM
            {self.schema}.PH_D_Ontology ont
            JOIN {self.schema}.PH_D_Ontology_Concept_Alias ontc ON (
                ont.population_id = ontc.population_id
                AND ont.concept_id = ontc.concept_id
                AND ont.context_id = ontc.context_id
                AND ont.context_id = '{self.context_id}'
                AND ont.population_id = '{self.population_id}'
            )
        """
        q += where
        return q
    
    def code_oid_query(self, code_oids):
        """
        Returns a query to search the ontology tables for specific code_oids.
        
        Parameters
        ----------
        code_oid: list
            list of code_oids to query
        """
        
        q = f"""
        SELECT DISTINCT
            ont.population_id,
            ont.code_oid,
            ont.code_system_id,
            ontc.alias
        FROM
            {self.schema}.PH_D_Ontology ont
            JOIN {self.schema}.PH_D_Ontology_Concept_Alias ontc ON (
                ont.population_id = ontc.population_id
                AND ont.concept_id = ontc.concept_id
                AND ont.context_id = ontc.context_id
                AND ont.context_id = '{self.context_id}'
                AND ont.population_id = '{self.population_id}'
                AND ont.code_oid {equal_in(code_oids)}
            )
        """
        return q
    
    def synapse_attribution_query(self):
        """
        Return attribution query with ref_record_type for attribution details.
        """
        
        q = f"""
        SELECT DISTINCT
            attr.empi_id,
            attr.prsnl_id,
            comp.ref_record_type,
            comp.source_id
        FROM
            {self.schema}.PH_F_Attribution attr
            JOIN {self.schema}.PH_V_Attribution_Component_Data_Point comp ON (
                attr.empi_id = comp.empi_id
                AND attr.prsnl_id = comp.prsnl_id
            )
        """
        return q
    
    def manually_unattributed_query(self):
        """
        Return the manual attribution query.
        """

        q = f"""
        SELECT DISTINCT
            personnel_id || empi_id
        FROM
            {self.schema}.PH_F_Manual_Attribution
        WHERE
            attributed = False
        """
        return q
    
    def attribution_dt_tm_query(self):
        """
        Returns a query with the most recent attribution update datetime.
        """
        
        q = f"""
        SELECT DISTINCT
            MAX(attribution_end_date) AS max_attribution_dt_tm
        FROM
            {self.schema}.PH_F_Person_Attribution_Outcome_Changes
        """
        return q
    
    def lifetime_emr_query(self, data_partitions, primary_concepts, date):
        """
        Returns a query for lifetime emr.
        
        Parameters
        ----------
        data_partitions: list
            list of data partitions
        primary_concepts: alias_query
            alias_query() containing primary concepts
        date: str
            date to consider for begin and end date logic (e.g. begin <= date <= end)
        """
        
        q = f"""
        SELECT DISTINCT
            ppr.population_id,
            ppr.empi_id,
            ppr.provider_id,
            ppr.begin_date_id,
            ppr.begin_dt_tm,
            ppr.end_date_id,
            ppr.end_dt_tm,
            'PERSON_PROVIDER_RELATIONSHIP' AS ref_record_type,
            ppr.source_id
        FROM
            {self.schema}.PH_F_Person_Personnel_Reltn ppr
            JOIN ({primary_concepts}) pc ON (
                ppr.population_id = pc.population_id
                AND ppr.relationship_type_code = pc.code_oid
                AND ppr.relationship_type_coding_system_id = pc.code_system_id
                AND (
                    ppr.begin_dt_tm IS NULL
                    OR ppr.begin_dt_tm <= '{date}'
                )
                AND (
                    ppr.end_dt_tm IS NULL
                    OR ppr.end_dt_tm >= '{date}'
                )
                AND (
                    ppr.begin_dt_tm IS NULL
                    OR ppr.end_dt_tm IS NULL
                    OR ppr.begin_dt_tm <= ppr.end_dt_tm
                )
                AND SPLIT_PART(SPLIT_PART(ppr.ppr_id, ':', 2), '/', 1) {equal_in(data_partitions)}
                AND ppr.population_id = '{self.population_id}'
            )
        """
        return q
    
    def lifetime_enrollment_query(self, data_partitions, primary_concepts, date, lookback_type, lookback):
        """
        Returns a query for lifetime enrollment.
        
        Parameters
        ----------
        data_partitions: list
            list of data partitions
        primary_concepts: alias_query
            alias_query() containing primary concepts
        date: str
            date to consider for begin and end date logic (e.g. begin <= date <= end)
        lookack_type: str
            type of lookback
        lookback: int
            length of lookback period
        """
        
        q = f"""
        SELECT DISTINCT
            ppr.population_id,
            ppr.empi_id,
            ppr.provider_id,
            ppr.begin_date_id,
            ppr.begin_dt_tm,
            ppr.end_date_id,
            ppr.end_dt_tm,
            'PERSON_PROVIDER_RELATIONSHIP' AS ref_record_type,
            ppr.source_id
        FROM
            {self.schema}.PH_F_Person_Personnel_Reltn ppr
            JOIN ({primary_concepts}) pc ON (
                ppr.population_id = pc.population_id
                AND ppr.relationship_type_code = pc.code_oid
                AND ppr.relationship_type_coding_system_id = pc.code_system_id
                AND (
                    ppr.begin_dt_tm IS NULL
                    OR ppr.begin_dt_tm <= '{date}'
                )
                AND (
                    ppr.end_dt_tm IS NULL
                    OR ppr.end_dt_tm >= TIMESTAMPADD({lookback_type}, -{lookback}, '{date}')
                )
                AND (
                    ppr.begin_dt_tm IS NULL
                    OR ppr.end_dt_tm IS NULL
                    OR ppr.begin_dt_tm <= ppr.end_dt_tm
                )
                AND SPLIT_PART(SPLIT_PART(ppr.ppr_id, ':', 2), '/', 1) {equal_in(data_partitions)}
                AND ppr.population_id = '{self.population_id}'
            )
        """
        return q
    
    def visit_emr_query(self, data_partitions, primary_concepts, date, encounter_lookback, lookback_type, encounter_relationship = None, encounter_service = None):
        """
        Returns a query for visit emrs.
        
        Parameters
        ----------
        data_partitions: list
            list of data partitions
        primary_concepts: alias_query
            query for filtering concept aliases
        encounter_lookback: int
            how far to look back
        lookback_type:
            unit of look back period (e.g. MONTH, YEAR)
        encounter_relationship: alias_query
            query for filtering concept aliases
        """
        
        q = f"""
        SELECT DISTINCT
            enc.population_id,
            enc.empi_id,
            epr.provider_id,
            epr.begin_date_id,
            epr.begin_dt_tm,
            epr.end_date_id,
            epr.end_dt_tm,
            enc.service_date_id,
            enc.service_dt_tm,
            enc.discharge_date_id,
            enc.discharge_dt_tm,
            'ENCOUNTER' AS ref_record_type,
            enc.source_id
        FROM
            {self.schema}.PH_F_Encounter enc
            JOIN ({primary_concepts}) pc ON (
                enc.population_id = pc.population_id
                AND enc.encounter_type_code = pc.code_oid
                AND enc.encounter_type_coding_system_id = pc.code_system_id
            )
            JOIN {self.schema}.PH_F_Encounter_Personnel_Reltn epr ON (
                enc.population_id = epr.population_id
                AND enc.encounter_id = epr.encounter_id
                AND enc.empi_id = epr.empi_id
                AND enc.actual_arrival_dt_tm IS NOT NULL
                AND (
                    enc.actual_arrival_dt_tm <= enc.discharge_dt_tm
                    OR enc.discharge_dt_tm IS NULL
                )
                AND enc.service_dt_tm >= TIMESTAMPADD({lookback_type}, -{encounter_lookback}, '{date}')
                AND enc.service_dt_tm <= '{date}'
                AND SPLIT_PART(enc.encounter_id, ':', 1) {equal_in(data_partitions)}
            )"""
        if encounter_relationship is not None:
            q += f"""
            JOIN ({encounter_relationship}) er ON (
                epr.population_id = er.population_id
                AND epr.relationship_type_code = er.code_oid
                AND epr.relationship_type_coding_system_id = er.code_system_id
            )
        """
        if encounter_service is not None:
            q += f"""
                JOIN ({encounter_service}) es ON (
                    enc.population_id = es.population_id
                    AND enc.hospital_service_code = es.code_oid
                    AND enc.hospital_service_coding_system_id = es.code_system_id
                )
            """
        return q
    
    def visit_emr_max_dt_tm_query(self, visit_emr_query):
        """
        Returns visit_emr_query with max_dt_tm for each grouping
        
        Parameters
        ----------
        visit_emr_query: visit_emr_query
            the visit emr query
        """
        
        max_dt_tm = f"""
        SELECT DISTINCT
            ve.population_id,
            ve.empi_id,
            ve.provider_id,
            ve.service_date_id,
            MAX(ve.service_dt_tm) AS service_dt_tm
        FROM
            ({visit_emr_query}) ve
        GROUP BY
            ve.population_id,
            ve.empi_id,
            ve.provider_id,
            ve.service_date_id
        """
        q = f"""
        SELECT DISTINCT
            ve.*
        FROM
            ({visit_emr_query}) ve
            JOIN ({max_dt_tm}) mdt ON (
                ve.population_id = mdt.population_id
                AND ve.empi_id = mdt.empi_id
                AND ve.provider_id = mdt.provider_id
                AND ve.service_dt_tm = mdt.service_dt_tm
            )
        """
        return q
    
    def visit_claim_query(self, data_partitions, primary_concepts, date, procedure_lookback, lookback_type, claim_uid = True):
        """
        Returns a query for visit claims.
        
        Parameters
        ----------
        data_partitions: list
            list of data partitions
        primary_concepts: alias_query
            query for filtering concept aliases
        procedure_lookback: int
            how far to look back
        lookback_type:
            unit of look back period (e.g. MONTH, YEAR)
        claim_uid: bool (default True)
            whether or not to check if claim_uid is present
        """
        
        if claim_uid:
            claim = "IS NOT NULL"
        else:
            claim = "IS NULL"
        q = f"""
        SELECT DISTINCT
            proc.population_id,
            proc.empi_id,
            ppr.provider_id,
            proc.service_start_dt_tm,
            proc.service_start_date_id,
            'PROCEDURE' AS ref_record_type,
            proc.source_id
        FROM
            {self.schema}.PH_F_Procedure proc
            JOIN ({primary_concepts}) pc ON (
                proc.population_id = pc.population_id
                AND proc.procedure_code = pc.code_oid
                AND proc.procedure_coding_system_id = pc.code_system_id
            )
            JOIN {self.schema}.PH_F_Procedure_Personnel_Reltn ppr ON (
                proc.population_id = ppr.population_id
                AND proc.procedure_id = ppr.procedure_id
                AND proc.empi_id = ppr.empi_id
                AND (
                    proc.service_start_dt_tm <= proc.service_end_dt_tm
                    OR proc.service_end_dt_tm IS NULL
                )
                AND proc.service_start_dt_tm >= TIMESTAMPADD({lookback_type}, -{procedure_lookback}, '{date}')
                AND proc.service_start_dt_tm <= '{date}'
                AND SPLIT_PART(SPLIT_PART(proc.procedure_id, ':', 2), '/', 1) {equal_in(data_partitions)}
                AND proc.claim_uid {claim}
            )
        """
        return q
    
    def visit_claim_max_dt_tm_query(self, visit_claim_query):
        """
        Returns visit_claim_query with max_dt_tm for each grouping
        
        Parameters
        ----------
        visit_claim_query: visit_claim_query
            the visit claim query
        """
        
        max_dt_tm = f"""
        SELECT DISTINCT
            vc.population_id,
            vc.empi_id,
            vc.provider_id,
            vc.service_start_date_id,
            MAX(vc.service_start_dt_tm) AS service_start_dt_tm
        FROM
            ({visit_claim_query}) vc
        GROUP BY
            vc.population_id,
            vc.empi_id,
            vc.provider_id,
            vc.service_start_date_id
        """
        q = f"""
        SELECT DISTINCT
            vc.*
        FROM
            ({visit_claim_query}) vc
            JOIN ({max_dt_tm}) mdt ON (
                vc.population_id = mdt.population_id
                AND vc.empi_id = mdt.empi_id
                AND vc.provider_id = mdt.provider_id
                AND vc.service_start_dt_tm = mdt.service_start_dt_tm
            )
        """
        return q
    
    def most_visits_query(self, table, groupby=['empi_id', 'prsnl_id'], aggcol='service_date_id', schema='DATA_INSIGHTS_ANALYST', visit_count=None):
        """
        Writes a tie-breaker query based on the prsnl_id(s) with the most visits with an empi_id.

        Parameters
        ----------
        groupby : list (default ['empi_id', 'prsnl_id'])
            columns to group by
        aggcol : str (default 'service_date_id')
            column to count distinct
        table : str
            table (or subquery) to use
        schema : str (default 'DATA_INSIGHTS_ANALYST')
            schema to use if table is not a subquery (set as None if using subquery)
            if schema is None, default to alias 'a'
        visit_count : int (default None)
            minimum number of visits
        """

        # check for schema
        if schema is not None:
            schema += '.' + table

            # columns separated by newlines
            g_s = ',\n\t'.join(groupby)

            # aggcol
            a = aggcol

        # if schema is None, assume table is subquery
        else:
            schema = f"({table}) a"

            # columns separated by newlines
            g_s = ',\n\t'.join(['a.' + i for i in groupby])

            # add alias to aggcol
            a = 'a.' + aggcol

        # service count query
        s = f"""
        SELECT DISTINCT
            {g_s},
            COUNT(DISTINCT {a}) AS service_count
        FROM
            {schema}
        GROUP BY
            {g_s}
        """

        # check visit_count
        if visit_count is not None:
            s += f"""
            HAVING
                COUNT(DISTINCT {aggcol}) >= {visit_count}"""

        # create new groups
        g_m = ',\n'.join(['a.' + i for i in groupby[:-1]])

        # max service count query
        m = f"""
        SELECT DISTINCT
            {g_m},
            MAX(a.service_count) AS service_count
        FROM
            ({s}) a
        GROUP BY
            {g_m}
        """

        # write joins for a and s
        a_join_s = '\nAND '.join([f'a.{i} = s.{i}' for i in groupby])

        # write joins for s and m
        s_join_m = '\nAND '.join([f's.{i} = m.{i}' for i in groupby[:-1]])

        # final query
        q = f"""
        SELECT DISTINCT
            a.*
        FROM
            {schema} a
            JOIN ({s}) s ON (
                {a_join_s}
            )
            JOIN ({m}) m ON (
                {s_join_m}
                AND s.service_count = m.service_count
            )
        """

        return q
    
    def most_recent_query(self, table, groupby=['empi_id', 'prsnl_id'], aggcol='service_dt_tm', schema='DATA_INSIGHTS_ANALYST'):
        """
        Writes a tie-breaker query based on the prsnl_id(s) with the most recent visit with an empi_id.

        Parameters
        ----------
        groupby : list (default ['empi_id', 'prsnl_id'])
            columns to group by
        aggcol : str (default 'service_dt_tm')
            column to max
        table : str
            table (or subquery) to use
        schema : str (default 'DATA_INSIGHTS_ANALYST')
            schema to use if table is not a subquery
            if schema is None, default to alias 'a'
        """

        # check for schema
        if schema is not None:
            schema += '.' + table

            # columns separated by newlines
            g_s = ',\n\t'.join(groupby)

            # aggcol
            a = aggcol

        # if schema is None, assume table is subquery
        else:
            schema = f"({table}) a"

            # columns separated by newlines
            g_s = ',\n\t'.join(['a.' + i for i in groupby])

            # add alias to aggcol
            a = 'a.' + aggcol

        # max service query
        s = f"""
        SELECT DISTINCT
            {g_s},
            MAX({a}) AS max_service
        FROM
            {schema}
        GROUP BY
            {g_s}
        """

        # create new groups
        g_m = ',\n'.join(['a.' + i for i in groupby[:-1]])

        # max max service query
        m = f"""
        SELECT DISTINCT
            {g_m},
            MAX(a.max_service) AS max_service
        FROM
            ({s}) a
        GROUP BY
            {g_m}
        """

        # write joins for a and s
        a_join_s = '\nAND '.join([f'a.{i} = s.{i}' for i in groupby])

        # write joins for s and m
        s_join_m = '\nAND '.join([f's.{i} = m.{i}' for i in groupby[:-1]])

        # final query
        q = f"""
        SELECT DISTINCT
            a.*
        FROM
            {schema} a
            JOIN ({s}) s ON (
                {a_join_s}
            )
            JOIN ({m}) m ON (
                {s_join_m}
                AND s.max_service = m.max_service
            )
        """

        return q
    
    def attribution_query(self, provider_type, relationships = None, visits = None, claims = None, visit_count = 1):
        """
        Returns the attribution query.
        
        Parameters
        ----------
        provider_type: query
            query for prsnl (e.g. pcp_query, specialist_query, etc.)
        relationships: query (default None)
            relationship query to be used
        visits: query
            visit query to be used
        claims: query
            claim query to be used
        visit_count: int
            minimum number of visits to be attributed
        soft_ppr_concept: list
            list of soft ppr concepts or encounter relationship concepts
        """
        
        if relationships:
            pass
        if visits:
            pass
        if claims:
            pass
        q = """
        SELECT DISTINCT
            *
        FROM
            
        """
        pass


# # Client Profiles

# In[ ]:


class client_profiles:
    """
    A set of functions needed to create client profiles for use in the ABC algorithm.
    """
    
    def user_activation():
        """
        Scrapes Operations data to get list of clients who have finished User Activation +6 months before a given date.
        
        Parameters
        ----------
        
        """
        pass
        
    def cerner_standard(username, password, user_activation, registries = ['comprehensiveadultwellness', 'adultwellness', 'diabetesmellitus', 'heartfailure', 'hypertension']):
        """
        Scrapes Synapse GitHub repository using the list of clients produced from `user_activation()` and a list of registries,
        then returns whether or not the client has the registry.
        
        Parameters
        ----------
        username: str
            the username used to login to github.cerner.com
        password: str
            the password used to login to gitub.cerner.com
        user_activation: list
            a custom list of schemas, or the list of client schemas from the user_activation function
        registries: list
            list of registries that will be included in the client profile
            (default ['comprehensiveadultwellness', 'adultwellness', 'diabetesmellitus', 'heartfailure', 'hypertension'])
        """
        username_value = username
        password_value = password
#         open a session
        session_requests = requests.session()
#         the login page
        login_url = 'https://github.cerner.com/session'
        result = session_requests.get(login_url)
#         did we make it to the website?
        if result.ok == False:
            raise Exception('Returned status of ' + str(result.status) + '. Please refer to https://httpstatuses.com/ for more information.')
        soup = BeautifulSoup(result.text, 'lxml')
#         get valid token
        for n in soup('input'):
            if ('log' or 'user') in n['name']:
                username_name = n['name']
            if 'pass' in n['name']:
                password_name = n['name']
            if 'token' in n['name']:
                token_name = n['name']
                token_value = n['value']
#         credentials
        payload = {
            username_name : username_value,
            password_name : password_value,
             token_name : token_value
        }
#         perform the login
        result = session_requests.post(login_url, data = payload, headers = dict(referer = login_url))
#         were we successful?
        if result.ok == False:
            raise Exception('Did not log in successfully. Returned status of ' + str(result.status) + '. Please refer to https://httpstatuses.com/ for more information.')
        profile = pd.DataFrame(index = [s.replace('_', '') for s in user_activation])
#         destination
        url = 'https://raw.github.cerner.com/Synapse/schema_name-rules/master/src/main/resources/schema_name/registry/clinical.clj'
        for s in profile.index.unique():
            for r in registries:
                result = session_requests.get(url.replace('schema_name', s.replace('_', '')).replace('registry', r), headers = dict(referer = url))
                profile.loc[s, r] = result.ok
        return profile


# # Benchmarks

# In[ ]:


class Benchmarks:
    """
    A set of functions to assist in creating the Achievable Benchmarks of Care (ABC) and QUALiTIERS.
    """
    
    def dream_dates(dream_quarter):
        """
        Returns a date range consisting of the first day of each month in the dream quarter.
        
        Parameters
        ----------
        dream_quarter: str
            quarter of interest starting with YYYY + Q1, Q2, Q3, or Q4 (e.g. 2018Q1)
        """
        # create dream quarter
        return pd.date_range(start = dream_quarter, periods = 3, freq = 'MS')
    
    def query(schema, dream_dates, scorable = True, measure_type = None, registries = None):
        """
        Query for pulling back data to be used for computing the ABC and QUALiTIERS.
        
        Parameters
        ----------
        schema : str
            the client schema
        dream_dates : pd.core.indexes.datetimes.DatetimeIndex
            the date range to limit data to
        scorable : bool (default True)
            whether the measures are scorable or not
            if True, return scorecard measures, else return dashboard history measures
        measure_type : str (default None)
            what types of measures to query (returns all by default)
            if "standard", return only cernerstandard measures,
            elif "custom", return only custom measures
        registries : list (default None)
            a list of registries to query based on the program written in the fully_qualified_name (returns all by default)
        Returns
        -------
        query : str
        """
        
        # validate parameter types
        # `schema`
        if type(schema) != str:
            raise TypeError(f"`schema` must be of type str. You supplied {type(schema)}")
        
        # `dream_dates`
        if type(dream_dates) != pd.core.indexes.datetimes.DatetimeIndex:
            raise TypeError(f"`dream_dates` must be of type pd.core.indexes.datetimes.DatetimeIndex. You supplied {type(dream_dates)}")
        
        # `scorable`
        if type(scorable) != bool:
            raise TypeError(f"`scorable` must be of type bool. You supplied {type(scorable)}")
        
        # `measure_type`
        if (measure_type is not None) and (type(measure_type) != str):
            raise TypeError(f"`measure_type` must be None or of type str. You supplied {type(measure_type)}")
        
        # `registries`
        if (registries is not None) and (type(registries) != list):
            raise TypeError(f"`registries must be None or of type list. You supplied {type(registries)}")
        
        # define min/max dates
        min_date = (dream_dates.min() - pd.offsets.Day(16)).date()
        max_date = (dream_dates.max() + pd.offsets.Day(14)).date()
        
        # scorable
        if scorable:
            
            # build scorable date_count query
            date_count_query = f"""
            SELECT DISTINCT
                fully_qualified_name,
                prsnl_id,
                COUNT(DISTINCT created_dt_tm) AS dt_count
            FROM
                {schema}.PH_F_Scorecard_Trending_Personnel_Point
            WHERE
                created_dt_tm BETWEEN DATE('{min_date}') AND DATE('{max_date}')
                AND denominator_value > 0
            GROUP BY
                fully_qualified_name,
                prsnl_id
            """
            
            # build scorable_query
            q = f"""
            SELECT DISTINCT
                population_id,
                created_dt_tm AS date,
                prsnl_id,
                fully_qualified_name,
                numerator_value AS numerator,
                denominator_value AS denominator
            FROM
                {schema}.PH_F_Scorecard_Trending_Personnel_Point
            WHERE
                created_dt_tm BETWEEN DATE('{min_date}') AND DATE('{max_date}')
                AND denominator_value > 0
            """
        # dashboard history
        else:
            
            # build dashboard history date_count_query
            date_count_query = f"""
            SELECT DISTINCT
                m.fully_qualified_name,
                st.dashboard_history_personnel_id AS prsnl_id,
                COUNT(DISTINCT st.created_dt_tm) AS dt_count
            FROM
                {schema}.PH_F_Dashboard_History_Trending_Registry_Measure m
                JOIN {schema}.PH_F_Dashboard_History_Trending_Registry r ON (
                    m.registry_trending_uid = r.registry_trending_uid
                    AND m.dashboard_history_uuid = r.dashboard_history_uuid
                )
                JOIN {schema}.PH_F_Dashboard_History_Trending_Registry_Mara_Risk_Score s ON (
                    r.registry_trending_uid = s.mara_risk_score_uuid
                    AND r.dashboard_history_uuid = s.dashboard_history_uuid
                )
                JOIN {schema}.PH_F_Dashboard_History_Trending_Statistic st ON (
                    r.dashboard_history_uuid = st.dashboard_history_uuid
                )
            WHERE
                st.created_dt_tm BETWEEN DATE('{min_date}') AND DATE('{max_date}')
                AND m.composite_percentage_met_count IS NOT NULL
                AND (m.composite_percentage_met_count + m.composite_percentage_not_met_count) > 0
                AND s.dashboard_registry_mara_risk_score_type = 'Attributed_Person_Registry_Statistics'
                AND st.dashboard_type = 'Personnel_Trending_Dashboard'
            GROUP BY
                m.fully_qualified_name,
                st.dashboard_history_personnel_id
            """
            
            # build dashboard history query
            q = f"""
            SELECT DISTINCT
                st.created_dt_tm AS date,
                m.fully_qualified_name,
                st.dashboard_history_personnel_id AS prsnl_id,
                m.composite_percentage_met_count AS numerator,
                (m.composite_percentage_met_count + m.composite_percentage_not_met_count) AS denominator
            FROM
                {schema}.PH_F_Dashboard_History_Trending_Registry_Measure m
                JOIN {schema}.PH_F_Dashboard_History_Trending_Registry r ON (
                    m.registry_trending_uid = r.registry_trending_uid
                    AND m.dashboard_history_uuid = r.dashboard_history_uuid
                )
                JOIN {schema}.PH_F_Dashboard_History_Trending_Registry_Mara_Risk_Score s ON (
                    r.registry_trending_uid = s.mara_risk_score_uuid
                    AND r.dashboard_history_UUID = s.dashboard_history_UUID
                )
                JOIN {schema}.PH_F_Dashboard_History_Trending_Statistic st ON (
                    r.dashboard_history_uuid = st.dashboard_history_uuid
                )
            WHERE
                st.created_dt_tm BETWEEN DATE('{min_date}') AND DATE('{max_date}')
                AND m.composite_percentage_met_count IS NOT NULL
                AND (m.composite_percentage_met_count + m.composite_percentage_not_met_count) > 0
                AND s.dashboard_registry_mara_risk_score_type = 'Attributed_Person_Registry_Statistics'
                AND st.dashboard_type = 'Personnel_Trending_Dashboard'
            """
            
        # check if measure_type given
        if type(measure_type) == str:
            
            # options
            mt = ['standard', 'custom']
            
            # check if measure_type is in options
            if measure_type not in mt:
                raise ValueError(f"`measure_type` must be in {mt}. You supplied '{measure_type}'")
            
            # standard
            elif measure_type == 'standard':
                
                # filter to standard measures
                date_count_query = f"""
                SELECT DISTINCT
                    a.*
                FROM
                    ({date_count_query}) a
                WHERE
                    SPLIT_PART(fully_qualified_name, '.', 1)  = 'cernerstandard'
                """
                
            # custom
            else:
                # filter to custom measures
                date_count_query = f"""
                SELECT DISTINCT
                    a.*
                FROM
                    ({date_count_query}) a
                WHERE
                    SPLIT_PART(fully_qualified_name, '.', 1) = '{schema.lower().replace('_', '')}'
                """
                
        # check registries
        if (type(registries) == list) and (len(registries) > 0):
            
            # check if measure_type was None
            if measure_type is None:
                date_count_query = f"""
                SELECT DISTINCT
                    a.*
                FROM
                    ({date_count_query}) a
                WHERE
                    SPLIT_PART(fully_qualified_name, '.', 2) {equal_in(registries)}
                """
            # if measure_type was not None...
            else:
                date_count_query += f"""
                AND SPLIT_PART(fully_qualified_name, '.', 2) {equal_in(registries)}
                """
        # build query
        q = f"""
        SELECT DISTINCT
            '{schema}' AS client,
            q.*
        FROM
            ({q}) q
            JOIN ({date_count_query}) d ON (
                q.fully_qualified_name = d.fully_qualified_name
                AND q.prsnl_id = d.prsnl_id
                AND d.dt_count >= 3
            )
        """
        return q
    
    def date_check(df, dates):
        """
        Ensures that all prsnl_ids have exactly len(`dates`) dates per fully_qualified_name.

        Parameters
        ----------
        df : pd.DataFrame()
            ABC dataframe containing, at the minimum, prsnl_id, fully_qualified_name, and date.
        dates : array-like
            the "dream_dates"

        Returns
        -------
        df : pd.DataFrame()
            The cleaned df
        """

        # make a dataframe containing the dates (no duplicates
        date_df = df.loc[:, [
            'date'
        ]].drop_duplicates().copy()

        # insert a column for each date
        for d in dates:
            date_df.insert(date_df.shape[-1], d, d)

        # calculate the absolute difference between the given date and the dream_dates
        for d in dates:
            date_df.loc[:, d] = abs(date_df.loc[:, 'date'] - date_df.loc[:, d])

        # melt date_df to convert the columns into rows
        date_df = date_df.melt(id_vars = [
            'date'
        ], var_name = 'dream', value_name = 'diff')

        # group by the dream date, find the min difference, and merge with date_df
        date_df = date_df.merge(date_df.groupby('dream')[[
            'diff'
        ]].min()).drop_duplicates().drop([
            'dream',
            'diff'
        ], axis = 1)

        # merge date_df with df to filter the dates
        df = df.merge(date_df)

        # group by fully_qualified_name, prsnl_id, and date
        # take the max date
        # unstack the grouped date to make sure each column has the same value
        # drop any row with missing values
        # convert the index to a dataframe and merge with df
        df = df.merge(df.groupby([
            'fully_qualified_name',
            'prsnl_id',
            'date'
        ]).date.max().unstack('date').dropna().index.to_frame().reset_index(drop = True))

        return df
    
    def apf(x):
        """
        Calculates the APF and sorts the index in descending order.

        Parameters
        ----------
        x: DataFrame
            dataframe to perform function on
        """

        x['apf'] = (x.numerator + 1) / (x.denominator + 2)
        x1 = x.set_index('apf', append = True)
        x1.index = x1.index.droplevel(-2)
        return x1.groupby(level = x1.index.names).sum().sort_index(ascending = False)

    def population(x, ratio = .10):
        """
        Calculates population cutoff for the ABC.

        Parameters
        ----------
        x: DataFrame
            dataframe to perform function on
        ratio: float (default .10)
            the cutoff for the ABC. (e.g. .10 --> top 10%, .50 --> top 50%, etc.)
        """

        lvl = x.index.names[:-1]
        x1 = x.groupby(level = lvl).apply(np.cumsum)
        x2 = x1.reset_index().merge(x1.groupby(level = lvl).denominator.max().to_frame('md').reset_index())
        x2['cpp'] = x2.denominator / x2.md
        return x2.query('cpp >= {}'.format(ratio)).groupby(lvl).first()

    def abc(x):
        """
        Calculates the ABC.

        Parameters
        ----------
        x: DataFrame
            dataframe to perform function on
        """

        return (x.numerator / x.denominator).to_frame(name = 'abc')
    
    def tiers(s):
        """
        Takes a series of values between 0 and 1
        Returns points of inflection based a generated Gaussian KDE.

        Parameters
        ----------
        s : pd.Series or np.array
            the values to be used for computing the Gaussian KDE
        """

        # Compute the Gaussian KDE
        gkde = gaussian_kde(s)
        # Get 1000 x points between 0 and 1 for a full curve
        x = np.linspace(0, 1, 1000)
        # estimate y points based on x points
        y = gkde.pdf(x)
        # get relative max
        rel_max = argrelextrema(y, np.greater)[0]
        # Calculate the change in x
        dx = x[1] - x[0]
        # Calculate the slope
        dydx = np.gradient(y, dx)
        # Find the index for max points of inflection
        M = argrelextrema(dydx, np.greater)[0]
        # Find the index for min points of inflection
        m = argrelextrema(dydx, np.less)[0]
        # Filter the linear space by index
        Mv = x[M]
        mv = x[m]
        # Create dict
        d = {
            'M' : [],
            'P' : []
        }
        # check for relative max
        if len(rel_max) > 0:
            # if present, append to targets
            for i in rel_max:
                d['M'].append(x[i])
        # check for max inflection point
        if len(Mv) > 0:
            # if present, append to targets
            for j in Mv:
                d['P'].append(j)
        # check for min inflection point
        if len(mv) > 0:
            # if present, append to targets
            for l in mv:
                d['P'].append(l)
        # store relative max as array
        M = np.array(d['M'])
        # store y_data as series
        s = pd.Series(y)
        # filter x_data by index of max value of y_data
        MM = x[s.idxmax()]
        # store points of inflection as array
        poi = np.array(d['P'])
        # try to filter points of inflection less than MM, take max
        try:
            m_poi = poi[poi < MM].max()
        except ValueError:
            m_poi = float('nan')
        return m_poi
    
    def tier_plot(s, tiers, abcs, title):
        """
        Takes a series of measure met percentages and creates a histogram and kde plot.
        Then places vertical lines on tiers and abcs
        
        Parameters
        ----------
        s : pd.Series
            series of floats between 0.0 and 1.0
        tiers : float
            the lowest tier (usually created by `tiers()` method
        abcs : list
            list of abc targets (usually 50th and 90th)
        title: str
            the title of the figure
            
        Returns: dataframe
        """
        
        # define 5% bins
        bins = [i/100 for i in range(0, 101, 5)]
        # define figure and axes
        fig, ax = plt.subplots()
        ax1 = fig.gca()
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
        # plot KDE
        s.plot.kde(ax=ax)
        # plot histogram
        s.plot.hist(bins = bins, alpha = 0.5, ax=ax)
        # plot tiers
        ax.axvline(tiers, color = 'b', label = 'Tier 1')
        # plot abcs
        for i,abc in enumerate(sorted(abcs)):
            ax.axvline(abc, color = 'b', label = f'Tier {i+2}')
        # limit x-axis
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=10)
        return fig